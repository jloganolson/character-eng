"""Tests for the dashboard event collector and HTTP server."""

import json
import queue
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from character_eng.dashboard.events import DashboardEventCollector
from character_eng.dashboard.server import start_dashboard


class TestEventCollector:
    def test_push_and_get_all(self):
        c = DashboardEventCollector()
        c.push("test", {"msg": "hello"})
        c.push("test2", {"msg": "world"})
        events = c.get_all()
        assert len(events) == 2
        assert events[0]["type"] == "test"
        assert events[0]["data"]["msg"] == "hello"
        assert events[1]["seq"] == 2

    def test_subscribe_receives_events(self):
        c = DashboardEventCollector()
        q = c.subscribe()
        c.push("x", {"a": 1})
        event = q.get(timeout=1)
        assert event.type == "x"
        assert event.data["a"] == 1
        c.unsubscribe(q)

    def test_thread_safety(self):
        c = DashboardEventCollector()
        q = c.subscribe()

        def pusher():
            for i in range(100):
                c.push("t", {"i": i})

        threads = [threading.Thread(target=pusher) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        events = c.get_all()
        assert len(events) == 500
        # Sequence numbers should be unique and monotonic per push
        seqs = [e["seq"] for e in events]
        assert len(set(seqs)) == 500

    def test_shutdown_sends_none(self):
        c = DashboardEventCollector()
        q = c.subscribe()
        c.shutdown()
        event = q.get(timeout=1)
        assert event is None

    def test_reset_clears_history_but_keeps_sequence_progressing(self):
        c = DashboardEventCollector()
        c.push("a", {})
        c.push("b", {})
        c.reset()
        assert c.get_all() == []
        c.push("c", {})
        events = c.get_all()
        assert len(events) == 1
        assert events[0]["type"] == "c"
        assert events[0]["seq"] == 3


class TestDashboardServer:
    @pytest.fixture()
    def server(self):
        c = DashboardEventCollector()
        iq = queue.Queue()
        report_dir = Path("/tmp/character-eng-dashboard-test-reports")
        if report_dir.exists():
            for child in report_dir.iterdir():
                child.unlink()
        else:
            report_dir.mkdir(parents=True)
        thread, port = start_dashboard(c, iq, port=0, report_dir=report_dir)
        # Wait for server to be ready
        for _ in range(20):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
                break
            except Exception:
                time.sleep(0.05)
        yield c, iq, port, report_dir
        # Server thread is daemon, will die with test

    def test_root_returns_html(self, server):
        _, _, port, _ = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        body = resp.read().decode()
        assert "Character Dashboard" in body
        assert resp.status == 200

    def test_state_returns_json(self, server):
        c, _, port, _ = server
        c.push("test_event", {"key": "value"})
        time.sleep(0.05)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
        data = json.loads(resp.read())
        assert len(data) == 1
        assert data[0]["type"] == "test_event"

    def test_system_map_returns_html(self, server):
        _, _, port, _ = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/system-map.html")
        body = resp.read().decode()
        assert "Character Runtime & Design Map" in body
        assert resp.status == 200

    def test_stream_schema_returns_json(self, server):
        _, _, port, _ = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream-schema.json")
        body = json.loads(resp.read())
        assert "lane_order" in body
        assert body["event_lane_map"]["assistant_reply"] == "chat"
        assert resp.status == 200

    def test_send_injects_input(self, server):
        _, iq, port, _ = server
        body = json.dumps({"text": "hello world"}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/send",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        assert iq.get(timeout=1) == "hello world"

    def test_report_snapshot_saves_json(self, server):
        _, _, port, report_dir = server
        payload = {
            "session_id": "sess-123",
            "selected_event_key": "evt-9",
            "selected_bundle": {"event": {"type": "assistant_reply", "label": "Hi"}},
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/report-snapshot",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read())
        assert resp.status == 200
        assert data["ok"] is True
        saved = Path(data["path"])
        assert saved.exists()
        assert saved.parent == report_dir
        written = json.loads(saved.read_text())
        assert written["session_id"] == "sess-123"
