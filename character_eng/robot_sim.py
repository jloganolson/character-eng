from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict, dataclass, field


@dataclass
class RobotFaceState:
    gaze: str = "scene"
    gaze_type: str = "hold"
    expression: str = "neutral"
    source: str = "boot"
    updated_at: float = field(default_factory=time.time)


@dataclass
class WaveState:
    active: bool = False
    started_at: float = 0.0
    ends_at: float = 0.0
    source: str = ""


@dataclass
class FistBumpState:
    session_id: str = ""
    state: str = "idle"
    active: bool = False
    source: str = ""
    started_at: float = 0.0
    updated_at: float = 0.0
    expires_at: float = 0.0
    contact_at: float = 0.0
    timeout_at: float = 0.0
    retracted_at: float = 0.0
    note: str = ""


class RobotSimController:
    """Thread-safe state controller for the interactive robot simulation."""

    def __init__(
        self,
        *,
        wave_duration_s: float = 1.8,
        fistbump_timeout_s: float = 8.0,
        fistbump_result_hold_s: float = 1.8,
    ) -> None:
        self._lock = threading.RLock()
        self._wave_duration_s = max(0.05, float(wave_duration_s))
        self._fistbump_timeout_s = max(0.05, float(fistbump_timeout_s))
        self._fistbump_result_hold_s = max(0.05, float(fistbump_result_hold_s))
        self._face = RobotFaceState()
        self._wave = WaveState()
        self._fistbump = FistBumpState()

    def _refresh_locked(self, now: float) -> None:
        if self._wave.active and now >= self._wave.ends_at:
            self._wave.active = False

        fist = self._fistbump
        if fist.active and fist.expires_at and now >= fist.expires_at:
            fist.active = False
            fist.state = "timed_out"
            fist.updated_at = now
            fist.timeout_at = now
            fist.retracted_at = now + self._fistbump_result_hold_s
            fist.note = "No bump detected before timeout."

        if fist.state in {"contacted", "timed_out", "cancelled"} and fist.retracted_at and now >= fist.retracted_at:
            self._fistbump = FistBumpState()

    def set_face(
        self,
        *,
        gaze: str,
        gaze_type: str,
        expression: str,
        source: str = "runtime",
    ) -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            self._face = RobotFaceState(
                gaze=str(gaze or "scene").strip() or "scene",
                gaze_type="glance" if str(gaze_type or "").strip().lower() == "glance" else "hold",
                expression=str(expression or "neutral").strip() or "neutral",
                source=str(source or "runtime").strip() or "runtime",
                updated_at=now,
            )
            return self._snapshot_locked(now)

    def trigger_wave(self, *, source: str = "manual") -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            self._wave = WaveState(
                active=True,
                started_at=now,
                ends_at=now + self._wave_duration_s,
                source=str(source or "manual").strip() or "manual",
            )
            return self._snapshot_locked(now)

    def offer_fistbump(self, *, source: str = "manual") -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            session_id = f"fb-{uuid.uuid4().hex[:10]}"
            self._fistbump = FistBumpState(
                session_id=session_id,
                state="offered",
                active=True,
                source=str(source or "manual").strip() or "manual",
                started_at=now,
                updated_at=now,
                expires_at=now + self._fistbump_timeout_s,
                note="Arm extended. Waiting for contact.",
            )
            return self._snapshot_locked(now)

    def register_fistbump_contact(self, *, session_id: str = "", source: str = "manual") -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            fist = self._fistbump
            if not fist.session_id:
                raise RuntimeError("no active fist bump session")
            if session_id and session_id != fist.session_id:
                raise RuntimeError("stale fist bump session")
            fist.active = False
            fist.state = "contacted"
            fist.updated_at = now
            fist.contact_at = now
            fist.retracted_at = now + self._fistbump_result_hold_s
            fist.source = str(source or fist.source or "manual").strip() or "manual"
            fist.note = "Contact confirmed. Retracting."
            return self._snapshot_locked(now)

    def timeout_fistbump(self, *, session_id: str = "", source: str = "manual") -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            fist = self._fistbump
            if not fist.session_id:
                raise RuntimeError("no active fist bump session")
            if session_id and session_id != fist.session_id:
                raise RuntimeError("stale fist bump session")
            fist.active = False
            fist.state = "timed_out"
            fist.updated_at = now
            fist.timeout_at = now
            fist.retracted_at = now + self._fistbump_result_hold_s
            fist.source = str(source or fist.source or "manual").strip() or "manual"
            fist.note = "Fist bump timed out. Retracting."
            return self._snapshot_locked(now)

    def cancel_fistbump(self, *, session_id: str = "", source: str = "manual") -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            fist = self._fistbump
            if not fist.session_id:
                self._fistbump = FistBumpState()
                return self._snapshot_locked(now)
            if session_id and session_id != fist.session_id:
                raise RuntimeError("stale fist bump session")
            fist.active = False
            fist.state = "cancelled"
            fist.updated_at = now
            fist.retracted_at = now + min(0.6, self._fistbump_result_hold_s)
            fist.source = str(source or fist.source or "manual").strip() or "manual"
            fist.note = "Fist bump cancelled."
            return self._snapshot_locked(now)

    def snapshot(self) -> dict:
        now = time.time()
        with self._lock:
            self._refresh_locked(now)
            return self._snapshot_locked(now)

    def _snapshot_locked(self, now: float) -> dict:
        wave = asdict(self._wave)
        wave["seconds_left"] = round(max(self._wave.ends_at - now, 0.0), 3) if self._wave.active else 0.0

        fist = asdict(self._fistbump)
        fist["seconds_left"] = round(max(self._fistbump.expires_at - now, 0.0), 3) if self._fistbump.active else 0.0
        fist["can_bump"] = bool(self._fistbump.active and self._fistbump.state == "offered")

        if self._fistbump.state == "offered":
            headline = "Fist bump offered"
        elif self._wave.active:
            headline = "Wave in progress"
        else:
            headline = "Idle"

        return {
            "ok": True,
            "face": asdict(self._face),
            "wave": wave,
            "fistbump": fist,
            "headline": headline,
            "updated_at": now,
        }
