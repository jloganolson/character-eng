"""Person re-identification with SAM3 crops + torchreid ResNet50.

Pipeline:
  1. SAM3 "person" prompt -> detect all person instances (masks + bboxes)
  2. Crop each person -> torchreid ResNet50 -> ReID embedding -> identity
  3. Extra SAM3 prompts (user-defined) run in same encode pass
"""

from __future__ import annotations

import threading
import time

import cv2
import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Identity matching (body embeddings)
# ---------------------------------------------------------------------------
class PersonIdentityTracker:
    """Track person identities using ReID embeddings with cosine similarity + EMA."""

    def __init__(self, threshold: float = 0.5, ema_alpha: float = 0.8):
        self._threshold = threshold
        self._ema_alpha = ema_alpha
        self._identities: list[dict] = []
        self._next_id = 1

    def match(self, embedding: np.ndarray) -> str:
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)

        best_score = -1.0
        best_idx = -1
        for i, ident in enumerate(self._identities):
            score = float(np.dot(embedding, ident["embedding"]))
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score >= self._threshold and best_idx >= 0:
            ident = self._identities[best_idx]
            ident["embedding"] = (
                self._ema_alpha * ident["embedding"]
                + (1 - self._ema_alpha) * embedding
            )
            ident["embedding"] /= np.linalg.norm(ident["embedding"]) + 1e-8
            ident["last_seen"] = time.time()
            return ident["label"]

        label = f"Person {self._next_id}"
        self._next_id += 1
        self._identities.append({
            "embedding": embedding.copy(),
            "label": label,
            "last_seen": time.time(),
        })
        return label

    @property
    def roster(self) -> list[dict]:
        return [{"label": i["label"]} for i in self._identities]


# ---------------------------------------------------------------------------
# ReID feature extractor (torchreid ResNet50)
# ---------------------------------------------------------------------------
class ReIDExtractor:
    """Extract person re-identification embeddings using torchreid ResNet50."""

    def __init__(self, device: str = "cuda", dtype=None):
        self._model = None
        self._transform = None
        self._device = device
        self._dtype = dtype

    def load(self):
        import torchreid
        from torchvision import transforms

        self._model = torchreid.models.build_model(
            name="resnet50",
            num_classes=751,  # Market-1501 pretrained classes
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


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------
class PersonTracker:
    """Background person tracking using SAM3 + torchreid ResNet50 ReID.

    Args:
        cam: WebcamCapture instance.
        sam3_getter: callable returning (sam3_model, sam3_processor) or (None, None).
    """

    def __init__(self, cam, sam3_getter=None, device: str = "cuda", dtype=None):
        self._cam = cam
        self._sam3_getter = sam3_getter or (lambda: (None, None))
        self._device = device
        self._dtype = dtype
        self._reid: ReIDExtractor | None = None
        self._tracker = PersonIdentityTracker()
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

                # 1. SAM3: encode image ONCE, run all prompts
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

                # 2. ReID on person crops
                for person in persons:
                    x, y, w, h = person["bbox"]
                    fh, fw = frame.shape[:2]
                    x, y = max(0, x), max(0, y)
                    w, h = min(w, fw - x), min(h, fh - y)
                    if w < 10 or h < 10:
                        continue
                    emb = self._reid.extract(frame[y:y + h, x:x + w])
                    if emb is not None:
                        person["identity"] = self._tracker.match(emb)

                t_reid = time.perf_counter() - t0 - t_sam3

                # 3. Cache results
                results = []
                masks = []
                for p in persons:
                    results.append({
                        "bbox": p["bbox"],
                        "identity": p.get("identity", "Unknown"),
                        "confidence": p.get("confidence", 0.0),
                    })
                    masks.append(p.get("mask"))

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
                    }

            except Exception as e:
                print(f"PersonTracker error: {e}", flush=True)

            time.sleep(0.3)

    # ---- SAM3 unified (encode once, run all prompts) ----

    def _run_sam3_all(
        self, frame_bgr: np.ndarray, prompts: list[str]
    ) -> dict[str, list[dict]]:
        sam3_model, sam3_proc = self._sam3_getter()
        if sam3_model is None or sam3_proc is None:
            return {}

        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        img_inputs = sam3_proc(images=pil, return_tensors="pt").to(self._device)
        # Cast pixel values to match SAM3 model dtype (e.g. bfloat16)
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
                outputs, threshold=0.3, mask_threshold=0.5,
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
                            int(xs.min()), int(ys.min()),
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

    # ---- public API ----

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

        # Person masks
        for i, mask in enumerate(masks):
            if mask is None or mask.shape[:2] != out.shape[:2]:
                continue
            color = self.MASK_COLORS[i % len(self.MASK_COLORS)]
            overlay = np.zeros_like(out)
            overlay[mask] = color
            out = cv2.addWeighted(out, 1.0, overlay, 0.35, 0)

        # Extra SAM3 detections
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

        # Person boxes + labels
        for i, face in enumerate(faces):
            x, y, bw, bh = face["bbox"]
            color = (0, 165, 255)
            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)
            label = face["identity"]
            conf = face.get("confidence", 0)
            if conf:
                label += f" {conf:.0%}"
            cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        # Stats HUD
        if timing:
            prompts_str = timing.get("prompts", "person")
            hud_lines = [
                f"PersonTracker | resnet50 | [{prompts_str}]",
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
        return self._tracker.roster
