"""DashboardEventCollector — thread-safe event collection with SSE fan-out."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import asdict, dataclass, field


@dataclass
class DashboardEvent:
    type: str
    data: dict
    timestamp: float = field(default_factory=time.time)
    seq: int = 0

    def to_json(self) -> str:
        return json.dumps({"type": self.type, "data": self.data,
                           "timestamp": self.timestamp, "seq": self.seq})


class DashboardEventCollector:
    """Collects events pushed from the chat loop and fans them out to SSE subscribers."""

    def __init__(self):
        self._events: list[DashboardEvent] = []
        self._lock = threading.Lock()
        self._seq = 0
        self._subscribers: list[queue.Queue] = []

    def push(self, event_type: str, data: dict) -> None:
        with self._lock:
            self._seq += 1
            event = DashboardEvent(type=event_type, data=data, seq=self._seq)
            self._events.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def get_all(self) -> list[dict]:
        with self._lock:
            return [{"type": e.type, "data": e.data,
                     "timestamp": e.timestamp, "seq": e.seq}
                    for e in self._events]

    def reset(self) -> None:
        with self._lock:
            self._events.clear()

    def shutdown(self) -> None:
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(None)
                except queue.Full:
                    pass
