"""Optional rust-eyes runtime bridge for driving expressions and gaze."""

from __future__ import annotations

import json
import threading
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_EXPRESSION_MAP_PATH = Path(__file__).resolve().parent / "rust_eyes_expression_map.toml"


class RustEyesBridgeError(RuntimeError):
    """Raised when the rust-eyes bridge cannot resolve or send a runtime update."""


@dataclass(frozen=True)
class RustEyesFaceUpdate:
    gaze: str
    gaze_type: str
    expression: str
    source: str = ""


def _normalize_label(value: str) -> str:
    return " ".join(str(value or "").strip().lower().replace("_", " ").split())


def _default_expression_map_path(path: str | Path | None) -> Path:
    if path:
        return Path(path).expanduser().resolve()
    return DEFAULT_EXPRESSION_MAP_PATH


class RustEyesRuntimeClient:
    def __init__(
        self,
        *,
        base_url: str,
        owner: str = "character-eng",
        ttl: float = 1.0,
        expression_map_path: str | Path | None = None,
        timeout_s: float = 0.5,
    ) -> None:
        self.base_url = str(base_url or "").rstrip("/")
        self.owner = str(owner or "character-eng").strip() or "character-eng"
        self.ttl = max(0.05, float(ttl))
        self.expression_map_path = _default_expression_map_path(expression_map_path)
        self.timeout_s = max(0.05, float(timeout_s))
        self._lock = threading.RLock()
        self._expression_map: dict[str, str] | None = None
        self._expression_map_mtime_ns: int | None = None
        self._available_expression_keys: set[str] | None = None

    def _request_json(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        raw = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=raw,
            headers={"Content-Type": "application/json"},
            method="GET" if payload is None else "POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read().decode("utf-8"))

    def _load_expression_map_locked(self) -> dict[str, str]:
        try:
            stat = self.expression_map_path.stat()
        except FileNotFoundError as exc:
            raise RustEyesBridgeError(
                f"rust-eyes expression map not found: {self.expression_map_path}"
            ) from exc

        if self._expression_map is not None and self._expression_map_mtime_ns == stat.st_mtime_ns:
            return self._expression_map

        try:
            payload = tomllib.loads(self.expression_map_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RustEyesBridgeError(
                f"failed to parse rust-eyes expression map: {self.expression_map_path}: {exc}"
            ) from exc

        mapping_data = payload.get("expressions")
        if not isinstance(mapping_data, dict):
            raise RustEyesBridgeError(
                f"rust-eyes expression map must define an [expressions] table: {self.expression_map_path}"
            )

        mapping: dict[str, str] = {}
        for raw_name, raw_key in mapping_data.items():
            name = _normalize_label(str(raw_name or ""))
            if not name:
                continue
            key = str(raw_key or "").strip()
            if not key:
                continue
            mapping[name] = key

        self._expression_map = mapping
        self._expression_map_mtime_ns = stat.st_mtime_ns
        return mapping

    def _available_expression_keys_locked(self) -> set[str]:
        if self._available_expression_keys is None:
            try:
                payload = self._request_json("/api/runtime/expressions")
            except urllib.error.URLError as exc:
                raise RustEyesBridgeError(
                    f"failed to reach rust-eyes runtime expressions API at {self.base_url}/api/runtime/expressions: {exc}"
                ) from exc
            keys = payload.get("expressions")
            if not isinstance(keys, list):
                raise RustEyesBridgeError(
                    f"unexpected rust-eyes runtime expressions payload from {self.base_url}/api/runtime/expressions"
                )
            self._available_expression_keys = {str(item).strip() for item in keys if str(item).strip()}
        return self._available_expression_keys

    def resolve_expression_key(self, expression: str) -> str:
        normalized = _normalize_label(expression or "")
        if not normalized:
            raise RustEyesBridgeError(
                f"missing expression name; update {self.expression_map_path} [expressions] with the authored rust-eyes keys"
            )
        with self._lock:
            mapping = self._load_expression_map_locked()
            key = mapping.get(normalized, "").strip()
            if not key:
                raise RustEyesBridgeError(
                    f"no rust-eyes expression mapping for '{normalized}'; update {self.expression_map_path}"
                )
            available = self._available_expression_keys_locked()
            if key not in available:
                raise RustEyesBridgeError(
                    f"mapped rust-eyes expression key '{key}' for '{normalized}' is not available; "
                    f"update {self.expression_map_path} or author that key in rust-eyes"
                )
            return key

    @staticmethod
    def _map_gaze(gaze: str, gaze_type: str) -> dict[str, float]:
        normalized = _normalize_label(gaze or "")
        look_x = 0.0
        look_y = 0.0

        if any(token in normalized for token in ("left", "stage left")):
            look_x = -0.28
        elif any(token in normalized for token in ("right", "stage right")):
            look_x = 0.28

        if any(token in normalized for token in ("up", "high", "ceiling")):
            look_y = 0.2
        elif any(token in normalized for token in ("down", "low", "floor", "table")):
            look_y = -0.18

        # Glances read better with slightly more offset than steady holds.
        if str(gaze_type or "").strip().lower() == "glance":
            look_x *= 1.15
            look_y *= 1.15

        return {
            "lookX": round(look_x, 4),
            "lookY": round(look_y, 4),
            "convergence": 0.0,
        }

    def push_face(self, update: RustEyesFaceUpdate) -> dict[str, Any]:
        expression_key = self.resolve_expression_key(update.expression)
        payload = {
            "owner": self.owner,
            "ttl": self.ttl,
            "expressionKey": expression_key,
            "gaze": self._map_gaze(update.gaze, update.gaze_type),
        }
        try:
            return self._request_json("/api/runtime/update", payload=payload)
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RustEyesBridgeError(
                f"rust-eyes runtime update failed ({exc.code}) at {self.base_url}/api/runtime/update: {body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RustEyesBridgeError(
                f"failed to reach rust-eyes runtime at {self.base_url}/api/runtime/update: {exc}"
            ) from exc
