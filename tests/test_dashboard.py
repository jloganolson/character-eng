"""Tests for the dashboard event collector and HTTP server."""

import json
import queue
import threading
import time
import urllib.request

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


class TestDashboardServer:
    @pytest.fixture()
    def server(self):
        c = DashboardEventCollector()
        iq = queue.Queue()
        thread, port = start_dashboard(c, iq, port=0)
        # Wait for server to be ready
        for _ in range(20):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
                break
            except Exception:
                time.sleep(0.05)
        yield c, iq, port
        # Server thread is daemon, will die with test

    def test_root_returns_html(self, server):
        _, _, port = server
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/")
        body = resp.read().decode()
        assert "Character Dashboard" in body
        assert resp.status == 200

    def test_state_returns_json(self, server):
        c, _, port = server
        c.push("test_event", {"key": "value"})
        time.sleep(0.05)
        resp = urllib.request.urlopen(f"http://127.0.0.1:{port}/state")
        data = json.loads(resp.read())
        assert len(data) == 1
        assert data[0]["type"] == "test_event"

    def test_send_injects_input(self, server):
        _, iq, port = server
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
