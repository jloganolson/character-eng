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
from character_eng.history import HistoryService
from character_eng.person import PeopleState
from character_eng.world import Goals, Script, WorldState


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
        history_root = Path("/tmp/character-eng-dashboard-test-history")
        if history_root.exists():
            for child in history_root.rglob("*"):
                if child.is_file():
                    child.unlink()
            for child in sorted(history_root.glob("*"), reverse=True):
                if child.is_dir():
                    for nested in sorted(child.rglob("*"), reverse=True):
                        if nested.is_dir():
                            nested.rmdir()
                    child.rmdir()
        history = HistoryService(root=history_root, free_warning_gib=0.0)
        archive = history.start_session(session_id="sess-test", character="greg", model="test")
        archive.capture_checkpoint(
            label="start",
            character="greg",
            session_snapshot={"system_prompt": "system", "messages": [], "tagged_system_indices": {}},
            world=WorldState(),
            people=PeopleState(),
            scenario=None,
            script=Script(),
            goals=Goals(),
            log_entries=[],
            context_version=0,
            had_user_input=False,
        )
        thread, port = start_dashboard(c, iq, port=0, report_dir=report_dir, history_api=history)
        # Wait for server to be ready
        for _ in range(20):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
                break
            except Exception:
                time.sleep(0.05)
        yield c, iq, port, report_dir, history
        # Server thread is daemon, will die with test

    def test_root_returns_html(self, server):
        _, _, port, _, _ = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        body = resp.read().decode()
        assert "Character Dashboard" in body
        assert resp.status == 200

    def test_state_returns_json(self, server):
        c, _, port, _, _ = server
        c.push("test_event", {"key": "value"})
        time.sleep(0.05)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
        data = json.loads(resp.read())
        assert len(data) == 1
        assert data[0]["type"] == "test_event"

    def test_system_map_returns_html(self, server):
        _, _, port, _, _ = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/system-map.html")
        body = resp.read().decode()
        assert "Character Runtime & Design Map" in body
        assert resp.status == 200

    def test_stream_schema_returns_json(self, server):
        _, _, port, _, _ = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/stream-schema.json")
        body = json.loads(resp.read())
        assert "lane_order" in body
        assert body["event_lane_map"]["assistant_reply"] == "chat"
        assert resp.status == 200

    def test_send_injects_input(self, server):
        _, iq, port, _, _ = server
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
        _, _, port, report_dir, _ = server
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

    def test_history_state_and_actions(self, server):
        _, _, port, _, history = server

        state = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/state").read())
        assert state["current_session"]["session_id"] == "sess-test"
        listing = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/history/list").read())
        assert listing[0]["session_id"] == "sess-test"
        assert listing[0]["ref"] == "sess-test"

        checkpoint = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/history/checkpoint?session_id=sess-test").read()
        )
        assert checkpoint["ok"] is True
        assert checkpoint["session_id"] == "sess-test"

        annotation_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/history/annotation",
            data=json.dumps({"session_id": "sess-test", "note": "slow response"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        annotation_resp = json.loads(urllib.request.urlopen(annotation_req).read())
        assert annotation_resp["ok"] is True
        assert Path(annotation_resp["path"]).exists()

        annotations = json.loads(
            urllib.request.urlopen(f"http://127.0.0.1:{port}/history/annotations?session_id=sess-test").read()
        )
        assert len(annotations) == 1
        assert annotations[0]["note"] == "slow response"

        promote_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/history/promote",
            data=json.dumps({"session_id": "sess-test", "kind": "pinned"}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        promote_resp = json.loads(urllib.request.urlopen(promote_req).read())
        assert promote_resp["ok"] is True
        assert Path(promote_resp["path"]).exists()

        history.finalize_current()
        snippet_req = urllib.request.Request(
            f"http://127.0.0.1:{port}/history/snippet",
            data=json.dumps({"session_id": "sess-test", "title": "snapshot", "event_time_s": 0.0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        snippet_resp = json.loads(urllib.request.urlopen(snippet_req).read())
        assert snippet_resp["ok"] is True
        assert Path(snippet_resp["path"]).exists()
