"""Person tracking with SAM3 detections, ByteTrack association, and face-led identity.

Pipeline:
  1. SAM3 "person" prompt -> detect all person instances (masks + bboxes)
  2. ByteTrack associates person boxes across frames
  3. Overlapping face identity is authoritative for a person track
  4. torchreid embeddings are only used to recover an already-known identity
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from cjm_byte_track.core import BYTETracker
except Exception:
    BYTETracker = None


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    union = aw * ah + bw * bh - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_contains_point(bbox: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    x, y, w, h = bbox
    px, py = point
    return x <= px <= (x + w) and y <= py <= (y + h)


class PersonIdentityRegistry:
    """Store body embeddings only for already-known identities."""

    def __init__(self, threshold: float = 0.5, ema_alpha: float = 0.8):
        self._threshold = threshold
        self._ema_alpha = ema_alpha
        self._identities: dict[str, dict] = {}

    def remember(self, label: str, embedding: np.ndarray):
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        record = self._identities.get(label)
        if record is None:
            self._identities[label] = {
                "embedding": embedding.copy(),
                "last_seen": time.time(),
            }
            return

        record["embedding"] = (
            self._ema_alpha * record["embedding"] + (1 - self._ema_alpha) * embedding
        )
        record["embedding"] /= np.linalg.norm(record["embedding"]) + 1e-8
        record["last_seen"] = time.time()

    def lookup(self, embedding: np.ndarray) -> str | None:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_label = None
        best_score = -1.0
        for label, record in self._identities.items():
            score = float(np.dot(embedding, record["embedding"]))
            if score > best_score:
                best_score = score
                best_label = label

        if best_label is None or best_score < self._threshold:
            return None
        return best_label

    @property
    def roster(self) -> list[dict]:
        return [{"label": label} for label in sorted(self._identities)]


class ReIDExtractor:
    """Extract person re-identification embeddings using torchreid ResNet50."""

    def __init__(self, device: str = "cuda", dtype=None):
        self._model = None
        self._transform = None
        self._device = device
        # torchreid's pretrained ResNet path is stable in fp32/fp16, but the
        # bf16 path can fail with mixed input/weight dtypes on some CUDA stacks.
        self._dtype = None if dtype == torch.bfloat16 else dtype

    def load(self):
        import torchreid
        from torchvision import transforms

        self._model = torchreid.models.build_model(
            name="resnet50",
            num_classes=751,
            loss="softmax",
            pretrained=True,
        )
        self._model = self._model.to(self._device).eval()
        if self._dtype is not None:
            self._model = self._model.to(self._dtype)

        self._transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("ReID model (resnet50) loaded.", flush=True)

    def extract(self, person_crop_bgr: np.ndarray) -> np.ndarray | None:
        if self._model is None:
            return None
        rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        tensor = self._transform(rgb).unsqueeze(0).to(device=self._device, dtype=self._dtype)
        with torch.no_grad():
            features = self._model(tensor)
        feat = features.cpu().float().numpy().flatten()
        feat = feat / (np.linalg.norm(feat) + 1e-8)
        return feat


class PersonTracker:
    """Background person tracking using SAM3 + ByteTrack + face-led identity."""

    def __init__(self, cam, sam3_getter=None, face_getter=None, device: str = "cuda", dtype=None):
        self._cam = cam
        self._sam3_getter = sam3_getter or (lambda: (None, None))
        self._face_getter = face_getter or (lambda: [])
        self._device = device
        self._dtype = dtype
        self._reid: ReIDExtractor | None = None
        self._identity_registry = PersonIdentityRegistry()
        self._byte_tracker = (
            BYTETracker(track_thresh=0.35, track_buffer=30, match_thresh=0.8, frame_rate=3)
            if BYTETracker is not None
            else None
        )
        self._track_memory: dict[int, dict] = {}
        self._frame_index = 0
        self._lock = threading.Lock()
        self._faces: list[dict] = []
        self._masks: list[np.ndarray | None] = []
        self._extra_dets: list[dict] = []
        self._sam3_extra_prompts: str = ""
        self._timing: dict = {}
        self._enabled = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._status = "not_loaded"
        self._error = ""

    @property
    def status(self) -> str:
        return self._status

    @property
    def error(self) -> str:
        return self._error

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_sam3_prompts(self, prompts: str):
        self._sam3_extra_prompts = prompts

    def enable(self):
        if self._enabled:
            return
        if self._status == "not_loaded":
            self._status = "loading"
            threading.Thread(target=self._load_models, daemon=True).start()
        self._enabled = True
        self._start_thread()

    def disable(self):
        self._enabled = False
        self._stop_thread()
        with self._lock:
            self._faces = []
            self._masks = []
            self._timing = {}

    def _load_models(self):
        try:
            self._reid = ReIDExtractor(device=self._device, dtype=self._dtype)
            self._reid.load()
        except Exception as e:
            print(f"ReID load error: {e}", flush=True)
            self._status = "error"
            self._error = f"ReID: {e}"
            return
        self._status = "ready"
        print("PersonTracker ready (resnet50).", flush=True)

    def _start_thread(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _stop_thread(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

    @staticmethod
    def _track_bbox(track) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = [int(v) for v in track.tlbr]
        return (x1, y1, max(0, x2 - x1), max(0, y2 - y1))

    def _track_person_matches(self, tracks: list, person_dets: list[dict]) -> list[tuple[int, dict]]:
        matches: list[tuple[int, dict]] = []
        if not tracks or not person_dets:
            return matches

        remaining = set(range(len(person_dets)))
        ordered_tracks = sorted(tracks, key=lambda track: float(getattr(track, "score", 0.0)), reverse=True)
        for track in ordered_tracks:
            track_bbox = self._track_bbox(track)
            best_idx = None
            best_iou = 0.0
            for idx in remaining:
                iou = _bbox_iou(track_bbox, person_dets[idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is None or best_iou < 0.3:
                continue
            remaining.remove(best_idx)
            matches.append((int(track.track_id), person_dets[best_idx]))
        return matches

    @staticmethod
    def _person_face_score(person_bbox: tuple[int, int, int, int], face: dict) -> float:
        face_bbox = tuple(face.get("bbox", (0, 0, 0, 0)))
        if len(face_bbox) != 4:
            return -1.0
        fx, fy, fw, fh = face_bbox
        center = (fx + fw / 2.0, fy + fh / 2.0)
        if not _bbox_contains_point(person_bbox, center):
            return -1.0

        px, py, pw, ph = person_bbox
        if pw <= 0 or ph <= 0:
            return -1.0

        expected_x = px + pw * 0.5
        expected_y = py + ph * 0.2
        dx = abs(center[0] - expected_x) / max(pw, 1)
        dy = abs(center[1] - expected_y) / max(ph, 1)
        conf = float(face.get("confidence", 0.0))
        return conf - dx - dy

    def _assign_faces_to_tracks(self, matches: list[tuple[int, dict]], faces: list[dict]) -> dict[int, dict]:
        assignments: dict[int, dict] = {}
        candidates: list[tuple[float, int, dict]] = []
        usable_faces = []
        for face in faces:
            identity = str(face.get("identity", "")).strip()
            if not identity or identity.lower() == "unknown":
                continue
            usable_faces.append(face)

        for track_id, det in matches:
            for face in usable_faces:
                score = self._person_face_score(det["bbox"], face)
                if score > 0.0:
                    candidates.append((score, track_id, face))

        used_tracks: set[int] = set()
        used_faces: set[int] = set()
        for score, track_id, face in sorted(candidates, key=lambda item: item[0], reverse=True):
            if track_id in used_tracks:
                continue
            face_key = id(face)
            if face_key in used_faces:
                continue
            assignments[track_id] = face
            used_tracks.add(track_id)
            used_faces.add(face_key)
        return assignments

    def _extract_body_embedding(self, frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray | None:
        if self._reid is None:
            return None
        x, y, w, h = bbox
        fh, fw = frame.shape[:2]
        x, y = max(0, x), max(0, y)
        w, h = min(w, fw - x), min(h, fh - y)
        if w < 24 or h < 48:
            return None
        return self._reid.extract(frame[y:y + h, x:x + w])

    def _resolve_identity(
        self,
        track_id: int,
        det: dict,
        frame: np.ndarray,
        face_assignments: dict[int, dict],
    ) -> tuple[str, str, np.ndarray | None]:
        memory = self._track_memory.setdefault(
            track_id,
            {
                "identity": f"Track {track_id}",
                "source": "track",
                "last_seen_frame": self._frame_index,
            },
        )
        memory["last_seen_frame"] = self._frame_index

        bound_face = face_assignments.get(track_id)
        embedding = None

        if bound_face is not None:
            identity = str(bound_face["identity"]).strip()
            memory["identity"] = identity
            memory["source"] = "face"
            embedding = self._extract_body_embedding(frame, det["bbox"])
            if embedding is not None:
                self._identity_registry.remember(identity, embedding)
            return identity, "face", embedding

        if memory.get("source") == "face":
            return str(memory["identity"]), "face-memory", None

        embedding = self._extract_body_embedding(frame, det["bbox"])
        if embedding is not None:
            recovered = self._identity_registry.lookup(embedding)
            if recovered is not None:
                memory["identity"] = recovered
                memory["source"] = "reid"
                return recovered, "reid", embedding

        identity = str(memory.get("identity", f"Track {track_id}"))
        if not identity:
            identity = f"Track {track_id}"
        memory["identity"] = identity
        memory["source"] = "track"
        return identity, "track", embedding

    def _prune_track_memory(self, active_track_ids: set[int]):
        stale_ids = [
            track_id
            for track_id, memory in self._track_memory.items()
            if track_id not in active_track_ids and (self._frame_index - int(memory.get("last_seen_frame", 0))) > 30
        ]
        for track_id in stale_ids:
            self._track_memory.pop(track_id, None)

    def _loop(self):
        while self._running and self._enabled:
            if self._reid is None or self._reid._model is None:
                time.sleep(0.5)
                continue

            frame = self._cam.get_frame()
            if frame is None:
                time.sleep(0.5)
                continue

            try:
                t0 = time.perf_counter()
                self._frame_index += 1

                prompts = ["person"]
                extra = self._sam3_extra_prompts.strip()
                if extra:
                    for p in extra.split(","):
                        p = p.strip()
                        if p and p.lower() != "person":
                            prompts.append(p)

                all_dets = self._run_sam3_all(frame, prompts)
                t_sam3 = time.perf_counter() - t0

                persons = all_dets.pop("person", [])
                extra_dets_list = []
                for prompt, dets in all_dets.items():
                    for d in dets:
                        d["prompt"] = prompt
                        extra_dets_list.append(d)

                detections = []
                for det in persons:
                    x, y, w, h = det["bbox"]
                    detections.append([x, y, x + w, y + h, det.get("confidence", 0.0)])
                det_array = (
                    np.asarray(detections, dtype=np.float32)
                    if detections
                    else np.empty((0, 5), dtype=np.float32)
                )

                if self._byte_tracker is None:
                    matches = [(index + 1, det) for index, det in enumerate(persons)]
                else:
                    tracks = self._byte_tracker.update(
                        det_array,
                        img_info=frame.shape[:2],
                        img_size=frame.shape[:2],
                    )
                    matches = self._track_person_matches(tracks, persons)

                faces = list(self._face_getter() or [])
                face_assignments = self._assign_faces_to_tracks(matches, faces)

                t_reid_start = time.perf_counter()
                results = []
                masks = []
                active_track_ids: set[int] = set()
                for track_id, det in matches:
                    active_track_ids.add(track_id)
                    identity, source, _embedding = self._resolve_identity(
                        track_id, det, frame, face_assignments
                    )
                    results.append({
                        "bbox": det["bbox"],
                        "track_id": track_id,
                        "identity": identity,
                        "identity_source": source,
                        "confidence": det.get("confidence", 0.0),
                    })
                    masks.append(det.get("mask"))
                t_reid = time.perf_counter() - t_reid_start
                self._prune_track_memory(active_track_ids)

                total = time.perf_counter() - t0
                with self._lock:
                    self._faces = results
                    self._masks = masks
                    self._extra_dets = extra_dets_list
                    self._timing = {
                        "sam3": round(t_sam3, 3),
                        "reid": round(t_reid, 3),
                        "total": round(total, 3),
                        "fps": round(1.0 / total, 1) if total > 0 else 0,
                        "n_persons": len(results),
                        "prompts": ", ".join(prompts),
                        "tracking": "bytetrack" if self._byte_tracker is not None else "none",
                    }
            except Exception as e:
                print(f"PersonTracker error: {e}", flush=True)

            time.sleep(0.3)

    def _run_sam3_all(
        self, frame_bgr: np.ndarray, prompts: list[str]
    ) -> dict[str, list[dict]]:
        sam3_model, sam3_proc = self._sam3_getter()
        if sam3_model is None or sam3_proc is None:
            return {}

        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if hasattr(sam3_proc, "set_image") and hasattr(sam3_proc, "set_text_prompt"):
            return self._run_native_sam3_all(sam3_proc, pil, prompts)

        img_inputs = sam3_proc(images=pil, return_tensors="pt").to(self._device)
        if self._dtype is not None:
            img_inputs["pixel_values"] = img_inputs["pixel_values"].to(self._dtype)

        with torch.no_grad():
            vision = sam3_model.get_vision_features(pixel_values=img_inputs.pixel_values)
        original_sizes = img_inputs.get("original_sizes").tolist()

        results: dict[str, list[dict]] = {}
        for prompt in prompts:
            text_inputs = sam3_proc(text=prompt, return_tensors="pt").to(self._device)
            with torch.no_grad():
                outputs = sam3_model(vision_embeds=vision, **text_inputs)

            post = sam3_proc.post_process_instance_segmentation(
                outputs,
                threshold=0.3,
                mask_threshold=0.5,
                target_sizes=original_sizes,
            )[0]

            dets = []
            masks = post.get("masks", [])
            boxes = post.get("boxes", [])
            scores = post.get("scores", [])

            for i in range(len(masks)):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < 0.3:
                    continue
                mask = masks[i]
                mask_np = (
                    mask.cpu().float().numpy().astype(bool)
                    if hasattr(mask, "cpu")
                    else np.array(mask, dtype=bool)
                )
                bbox = None
                if i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, "cpu"):
                        box = box.cpu().float().numpy()
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    bbox = (x1, y1, x2 - x1, y2 - y1)
                else:
                    ys, xs = np.where(mask_np)
                    if len(xs) > 0:
                        bbox = (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()) - int(xs.min()),
                            int(ys.max()) - int(ys.min()),
                        )
                if bbox and bbox[2] > 10 and bbox[3] > 10:
                    dets.append({
                        "bbox": bbox,
                        "mask": mask_np,
                        "confidence": round(score, 3),
                    })
            results[prompt] = dets

        return results

    def _run_native_sam3_all(
        self, sam3_proc, pil: Image.Image, prompts: list[str]
    ) -> dict[str, list[dict]]:
        state = sam3_proc.set_image(pil, state={})
        results: dict[str, list[dict]] = {}

        for prompt in prompts:
            prompt_state = sam3_proc.set_text_prompt(prompt, state)
            masks = prompt_state.get("masks", [])
            boxes = prompt_state.get("boxes", [])
            scores = prompt_state.get("scores", [])

            dets = []
            for i in range(len(masks)):
                score = float(scores[i]) if i < len(scores) else 0.0
                if score < 0.3:
                    continue
                mask = masks[i]
                mask_np = (
                    mask.detach().cpu().numpy()
                    if hasattr(mask, "detach")
                    else np.asarray(mask)
                )
                mask_np = np.squeeze(mask_np).astype(bool)

                bbox = None
                if i < len(boxes):
                    box = boxes[i]
                    if hasattr(box, "detach"):
                        box = box.detach().cpu().numpy()
                    x1, y1, x2, y2 = [int(v) for v in box[:4]]
                    bbox = (x1, y1, max(0, x2 - x1), max(0, y2 - y1))
                else:
                    ys, xs = np.where(mask_np)
                    if len(xs) > 0:
                        x1, x2 = int(xs.min()), int(xs.max())
                        y1, y2 = int(ys.min()), int(ys.max())
                        bbox = (x1, y1, x2 - x1, y2 - y1)
                if bbox is None:
                    continue
                dets.append({
                    "bbox": bbox,
                    "mask": mask_np,
                    "confidence": score,
                    "label": prompt,
                })
            results[prompt] = dets

        return results

    def get_faces(self) -> list[dict]:
        with self._lock:
            return list(self._faces)

    MASK_COLORS = [
        (0, 0, 255), (0, 165, 255), (0, 255, 255), (0, 255, 0),
        (255, 255, 0), (255, 0, 0), (255, 0, 255), (128, 0, 128),
    ]
    EXTRA_COLORS = [
        (255, 255, 0), (255, 0, 255), (128, 0, 128),
        (0, 128, 255), (128, 255, 0), (0, 128, 128),
    ]

    def annotate_frame(self, bgr: np.ndarray) -> np.ndarray:
        with self._lock:
            faces = list(self._faces)
            masks = list(self._masks)
            extra_dets = list(self._extra_dets)
            timing = dict(self._timing)

        out = bgr.copy()

        for i, mask in enumerate(masks):
            if mask is None or mask.shape[:2] != out.shape[:2]:
                continue
            color = self.MASK_COLORS[i % len(self.MASK_COLORS)]
            overlay = np.zeros_like(out)
            overlay[mask] = color
            out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

        for i, det in enumerate(extra_dets):
            color = self.EXTRA_COLORS[i % len(self.EXTRA_COLORS)]
            mask = det.get("mask")
            if mask is not None and mask.shape[:2] == out.shape[:2]:
                overlay = np.zeros_like(out)
                overlay[mask] = color
                out = cv2.addWeighted(out, 1.0, overlay, 0.3, 0)
            bbox = det.get("bbox")
            if bbox:
                x, y, bw, bh = bbox
                cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
                label = det.get("prompt", "?")
                conf = det.get("confidence", 0)
                if conf:
                    label += f" {conf:.0%}"
                cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        for face in faces:
            x, y, bw, bh = face["bbox"]
            color = (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            label = face["identity"]
            conf = face.get("confidence", 0)
            source = face.get("identity_source", "")
            if source and source not in {"track", "face-memory"}:
                label += f" [{source}]"
            if conf:
                label += f" {conf:.0%}"
            cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        if timing:
            prompts_str = timing.get("prompts", "person")
            hud_lines = [
                f"PersonTracker | {timing.get('tracking', 'none')} | [{prompts_str}]",
                f"SAM3: {timing.get('sam3', 0)*1000:.0f}ms  ReID: {timing.get('reid', 0)*1000:.0f}ms",
                f"Total: {timing.get('total', 0)*1000:.0f}ms  FPS: {timing.get('fps', 0):.1f}  Persons: {timing.get('n_persons', 0)}",
            ]
            y_offset = 20
            for line in hud_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (4, y_offset - th - 4), (8 + tw, y_offset + 4), (0, 0, 0), -1)
                cv2.putText(out, line, (6, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y_offset += th + 10

        return out

    def build_context_string(self) -> str:
        faces = self.get_faces()
        if not faces:
            return ""
        parts = [f["identity"] for f in faces]
        summary = "; ".join(parts)
        return f"[Person context: {len(faces)} person{'s' if len(faces) != 1 else ''}. {summary}.]"

    @property
    def identity_roster(self) -> list[dict]:
        active = {str(face["identity"]) for face in self.get_faces() if str(face.get("identity", "")).strip()}
        roster = {item["label"] for item in self._identity_registry.roster}
        labels = sorted(active | roster)
        return [{"label": label} for label in labels]
