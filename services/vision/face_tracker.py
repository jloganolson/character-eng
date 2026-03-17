"""InsightFace face tracking with identity persistence and gaze direction."""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np

try:
    from cjm_byte_track.core import BYTETracker
except Exception:
    BYTETracker = None


class IdentityTracker:
    """Track face identities across frames using cosine similarity on 512-dim embeddings."""

    def __init__(self, threshold: float = 0.4, ema_alpha: float = 0.8):
        self._threshold = threshold
        self._ema_alpha = ema_alpha
        self._identities: list[dict] = []
        self._next_id = 1

    def match(self, embedding: np.ndarray, age: int | None = None, gender: str | None = None) -> str:
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
            if age is not None:
                ident["age"] = age
            if gender is not None:
                ident["gender"] = gender
            return ident["label"]

        label = f"Person {self._next_id}"
        self._next_id += 1
        self._identities.append({
            "embedding": embedding.copy(),
            "label": label,
            "age": age,
            "gender": gender,
        })
        return label

    @property
    def roster(self) -> list[dict]:
        return [
            {"label": i["label"], "age": i["age"], "gender": i["gender"]}
            for i in self._identities
        ]


def _gaze_direction(yaw: float, pitch: float) -> str:
    """Convert yaw/pitch angles to a human-readable gaze direction."""
    if abs(yaw) < 15.0 and abs(pitch) < 10.0:
        return "at camera"
    parts = []
    if pitch < -10.0:
        parts.append("up")
    elif pitch > 10.0:
        parts.append("down")
    if yaw < -15.0:
        parts.append("left")
    elif yaw > 15.0:
        parts.append("right")
    return " ".join(parts) if parts else "at camera"


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


class FaceTracker:
    """Background face detection + identity tracking using InsightFace buffalo_s."""

    def __init__(self, cam, device: str = "cuda"):
        self._cam = cam
        self._device = device
        self._app = None  # lazy-loaded InsightFace model
        self._tracker = IdentityTracker()
        self._byte_tracker = (
            BYTETracker(track_thresh=0.35, track_buffer=30, match_thresh=0.8, frame_rate=10)
            if BYTETracker is not None
            else None
        )
        self._track_memory: dict[int, dict] = {}
        self._frame_index = 0
        self._lock = threading.Lock()
        self._faces: list[dict] = []
        self._timing: dict = {}
        self._enabled = False
        self._running = False
        self._thread: threading.Thread | None = None
        self._status = "not_loaded"  # not_loaded | loading | ready | error
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

    def enable(self):
        """Enable face tracking — lazy-loads model on first call."""
        if self._enabled:
            return
        if self._status == "not_loaded":
            self._status = "loading"
            threading.Thread(target=self._load_model, daemon=True).start()
        self._enabled = True
        self._start_thread()

    def disable(self):
        self._enabled = False
        self._stop_thread()
        with self._lock:
            self._faces = []

    def _load_model(self):
        try:
            from insightface.app import FaceAnalysis
            providers = []
            if self._device != "cpu":
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            self._app = FaceAnalysis(
                name="buffalo_s",
                providers=providers,
            )
            ctx_id = 0 if self._device != "cpu" else -1
            self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._status = "ready"
            print("InsightFace buffalo_s ready.", flush=True)
        except Exception as e:
            self._status = "error"
            self._error = str(e)
            print(f"InsightFace load error: {e}", flush=True)

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

    @staticmethod
    def _extract_face_data(face) -> dict:
        box = face.bbox.astype(int)
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1

        age = int(face.age) if hasattr(face, "age") else None
        gender = "M" if (hasattr(face, "gender") and face.gender == 1) else "F"
        confidence = float(face.det_score) if hasattr(face, "det_score") else 0.0

        head_pose = None
        looking_at_camera = False
        gaze_dir = "unknown"
        if hasattr(face, "pose") and face.pose is not None:
            pitch = float(face.pose[0])
            yaw = float(face.pose[1])
            roll = float(face.pose[2])
            head_pose = {"pitch": round(pitch, 1), "yaw": round(yaw, 1), "roll": round(roll, 1)}
            looking_at_camera = abs(yaw) < 15.0 and abs(pitch) < 10.0
            gaze_dir = _gaze_direction(yaw, pitch)

        return {
            "bbox": (int(x1), int(y1), int(bw), int(bh)),
            "age": age,
            "gender": gender,
            "confidence": round(confidence, 3),
            "head_pose": head_pose,
            "looking_at_camera": looking_at_camera,
            "gaze_direction": gaze_dir,
            "embedding": face.embedding if getattr(face, "embedding", None) is not None else None,
        }

    def _track_face_matches(self, tracks: list, face_data: list[dict]) -> list[tuple[int, dict]]:
        matches: list[tuple[int, dict]] = []
        if not tracks or not face_data:
            return matches

        remaining = set(range(len(face_data)))
        ordered_tracks = sorted(tracks, key=lambda track: float(getattr(track, "score", 0.0)), reverse=True)
        for track in ordered_tracks:
            track_bbox = self._track_bbox(track)
            best_idx = None
            best_iou = 0.0
            for idx in remaining:
                iou = _bbox_iou(track_bbox, face_data[idx]["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_idx is None or best_iou < 0.3:
                continue
            remaining.remove(best_idx)
            matches.append((int(track.track_id), face_data[best_idx]))
        return matches

    @staticmethod
    def _face_is_good_for_identity(face_data: dict, *, known_identity: bool) -> bool:
        x, y, w, h = face_data["bbox"]
        del x, y
        if face_data["embedding"] is None:
            return False
        if face_data["confidence"] < 0.45 or min(w, h) < 40:
            return False

        pose = face_data.get("head_pose")
        if pose is None:
            return face_data["confidence"] >= 0.8 and (w * h) >= 6400

        yaw = abs(float(pose.get("yaw", 999.0)))
        pitch = abs(float(pose.get("pitch", 999.0)))
        if known_identity:
            return yaw <= 20.0 and pitch <= 15.0
        return yaw <= 30.0 and pitch <= 20.0

    def _update_track_identity(self, track_id: int, face_data: dict) -> str:
        memory = self._track_memory.setdefault(
            track_id,
            {
                "identity": "Unknown",
                "age": face_data["age"],
                "gender": face_data["gender"],
                "last_seen_frame": self._frame_index,
                "last_identity_frame": -10_000,
            },
        )
        memory["last_seen_frame"] = self._frame_index

        known_identity = memory.get("identity", "Unknown") != "Unknown"
        if self._face_is_good_for_identity(face_data, known_identity=known_identity):
            cooldown = 10 if known_identity else 0
            last_identity_frame = int(memory.get("last_identity_frame", -10_000))
            if self._frame_index - last_identity_frame >= cooldown:
                memory["identity"] = self._tracker.match(
                    face_data["embedding"],
                    age=face_data["age"],
                    gender=face_data["gender"],
                )
                memory["age"] = face_data["age"]
                memory["gender"] = face_data["gender"]
                memory["last_identity_frame"] = self._frame_index

        if memory.get("age") is None and face_data["age"] is not None:
            memory["age"] = face_data["age"]
        if memory.get("gender") is None and face_data["gender"] is not None:
            memory["gender"] = face_data["gender"]

        return str(memory.get("identity", "Unknown"))

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
            if self._app is None:
                time.sleep(0.2)
                continue
            frame = self._cam.get_frame()
            if frame is None:
                time.sleep(0.1)
                continue
            try:
                t0 = time.perf_counter()
                faces_raw = self._app.get(frame)
                self._frame_index += 1
                face_data = [self._extract_face_data(face) for face in faces_raw]
                results = []

                if self._byte_tracker is None:
                    for detected in face_data:
                        identity = "Unknown"
                        if detected["embedding"] is not None:
                            identity = self._tracker.match(
                                detected["embedding"],
                                age=detected["age"],
                                gender=detected["gender"],
                            )
                        results.append({
                            "bbox": detected["bbox"],
                            "track_id": None,
                            "identity": identity,
                            "age": detected["age"],
                            "gender": detected["gender"],
                            "confidence": detected["confidence"],
                            "head_pose": detected["head_pose"],
                            "looking_at_camera": detected["looking_at_camera"],
                            "gaze_direction": detected["gaze_direction"],
                        })
                else:
                    detections = []
                    for detected in face_data:
                        x, y, bw, bh = detected["bbox"]
                        detections.append([x, y, x + bw, y + bh, detected["confidence"]])
                    det_array = (
                        np.asarray(detections, dtype=np.float32)
                        if detections
                        else np.empty((0, 5), dtype=np.float32)
                    )
                    tracks = self._byte_tracker.update(
                        det_array,
                        img_info=frame.shape[:2],
                        img_size=frame.shape[:2],
                    )
                    matches = self._track_face_matches(tracks, face_data)
                    active_track_ids = {track_id for track_id, _ in matches}
                    self._prune_track_memory(active_track_ids)

                    for track_id, detected in matches:
                        identity = self._update_track_identity(track_id, detected)
                        memory = self._track_memory.get(track_id, {})
                        results.append({
                            "bbox": detected["bbox"],
                            "track_id": track_id,
                            "identity": identity,
                            "age": memory.get("age", detected["age"]),
                            "gender": memory.get("gender", detected["gender"]),
                            "confidence": detected["confidence"],
                            "head_pose": detected["head_pose"],
                            "looking_at_camera": detected["looking_at_camera"],
                            "gaze_direction": detected["gaze_direction"],
                        })

                total = time.perf_counter() - t0
                with self._lock:
                    self._faces = results
                    self._timing = {
                        "total": round(total, 3),
                        "fps": round(1.0 / total, 1) if total > 0 else 0,
                        "n_faces": len(results),
                        "tracking": "bytetrack" if self._byte_tracker is not None else "embedding",
                    }
            except Exception as e:
                print(f"FaceTracker error: {e}", flush=True)

            time.sleep(0.1)  # ~10fps

    def get_faces(self) -> list[dict]:
        with self._lock:
            return list(self._faces)

    def annotate_frame(self, bgr: np.ndarray) -> np.ndarray:
        """Draw face boxes, identity labels, gaze arrows, and stats HUD on frame."""
        with self._lock:
            faces = list(self._faces)
            timing = dict(self._timing)

        out = bgr.copy()

        for face in faces:
            x, y, bw, bh = face["bbox"]
            looking = face.get("looking_at_camera", False)
            color = (0, 255, 0) if looking else (0, 165, 255)  # green or orange

            cv2.rectangle(out, (x, y), (x + bw, y + bh), color, 2)

            label = f'{face["identity"]} ~{face["age"]}{face["gender"]}'
            gaze = face.get("gaze_direction", "")
            if gaze:
                label += f" [{gaze}]"
            cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            # Draw gaze arrow from face center
            hp = face.get("head_pose")
            if hp:
                cx = x + bw // 2
                cy = y + bh // 2
                yaw_rad = math.radians(hp["yaw"])
                pitch_rad = math.radians(hp["pitch"])
                arrow_len = max(bw, bh) * 0.6
                dx = int(arrow_len * math.sin(yaw_rad))
                dy = int(arrow_len * math.sin(pitch_rad))
                cv2.arrowedLine(out, (cx, cy), (cx + dx, cy + dy), color, 2, tipLength=0.3)

        # Stats HUD (top-left)
        if timing:
            hud_lines = [
                f"FaceTracker | InsightFace buffalo_s | {timing.get('tracking', 'embedding')}",
                f"Total: {timing.get('total', 0)*1000:.0f}ms  FPS: {timing.get('fps', 0):.1f}  Faces: {timing.get('n_faces', 0)}",
            ]
            y_offset = 20
            for line in hud_lines:
                (tw, th), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(out, (4, y_offset - th - 4), (8 + tw, y_offset + 4), (0, 0, 0), -1)
                cv2.putText(out, line, (6, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                y_offset += th + 10

        return out

    def build_context_string(self) -> str:
        """Build a text string describing detected faces for VLM prompt injection."""
        faces = self.get_faces()
        if not faces:
            return ""
        parts = []
        for f in faces:
            desc = f'{f["identity"]}, age ~{f["age"]} {f["gender"]}'
            parts.append(desc)
        summary = "; ".join(parts)
        return f"[Face context: {len(faces)} face{'s' if len(faces) != 1 else ''}. {summary}.]"

    @property
    def identity_roster(self) -> list[dict]:
        return self._tracker.roster
