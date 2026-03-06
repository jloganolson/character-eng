"""InsightFace face tracking with identity persistence and gaze direction."""

from __future__ import annotations

import math
import threading
import time

import cv2
import numpy as np


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


class FaceTracker:
    """Background face detection + identity tracking using InsightFace buffalo_s."""

    def __init__(self, cam, device: str = "cuda"):
        self._cam = cam
        self._device = device
        self._app = None  # lazy-loaded InsightFace model
        self._tracker = IdentityTracker()
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
                results = []
                for face in faces_raw:
                    box = face.bbox.astype(int)
                    x1, y1, x2, y2 = box
                    bw, bh = x2 - x1, y2 - y1

                    age = int(face.age) if hasattr(face, "age") else None
                    gender = "M" if (hasattr(face, "gender") and face.gender == 1) else "F"
                    confidence = float(face.det_score) if hasattr(face, "det_score") else 0.0

                    identity = "Unknown"
                    if face.embedding is not None:
                        identity = self._tracker.match(face.embedding, age=age, gender=gender)

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

                    results.append({
                        "bbox": (int(x1), int(y1), int(bw), int(bh)),
                        "identity": identity,
                        "age": age,
                        "gender": gender,
                        "confidence": round(confidence, 3),
                        "head_pose": head_pose,
                        "looking_at_camera": looking_at_camera,
                        "gaze_direction": gaze_dir,
                    })

                total = time.perf_counter() - t0
                with self._lock:
                    self._faces = results
                    self._timing = {
                        "total": round(total, 3),
                        "fps": round(1.0 / total, 1) if total > 0 else 0,
                        "n_faces": len(results),
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
                f"FaceTracker | InsightFace buffalo_s",
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
            gaze = f.get("gaze_direction", "unknown")
            desc += f", looking {gaze}"
            parts.append(desc)
        summary = "; ".join(parts)
        return f"[Face context: {len(faces)} face{'s' if len(faces) != 1 else ''}. {summary}.]"

    @property
    def identity_roster(self) -> list[dict]:
        return self._tracker.roster
