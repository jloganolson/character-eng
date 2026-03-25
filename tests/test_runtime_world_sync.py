from unittest.mock import MagicMock
import threading

import character_eng.__main__ as app
from character_eng.person import PeopleState
from character_eng.world import WorldState


def test_push_world_people_state_includes_pending_and_people():
    world = WorldState(static=["Static fact"])
    world.add_fact("Dynamic fact")
    world.add_pending("A person approaches the stand")
    people = PeopleState()
    pid = people.get_or_create("Person 1")
    person = people.people[pid]
    person.presence = "present"
    person.add_fact("Standing near the stand")

    pushed = []
    previous_push = app._push
    try:
        app._push = lambda event_type, data: pushed.append((event_type, data))
        app._push_world_people_state(world, people)
    finally:
        app._push = previous_push

    event_types = [event_type for event_type, _ in pushed]
    assert "world_state" in event_types
    assert "people_state" in event_types

    world_payload = next(data for event_type, data in pushed if event_type == "world_state")
    assert world_payload["pending"] == ["A person approaches the stand"]
    assert any(item["text"] == "Dynamic fact" for item in world_payload["dynamic_facts"])

    people_payload = next(data for event_type, data in pushed if event_type == "people_state")
    assert len(people_payload["people"]) == 1
    assert people_payload["people"][0]["name"] == "Person 1"
    assert people_payload["people"][0]["presence"] == "present"


def test_start_reconcile_drains_pending_changes():
    world = WorldState()
    world.add_pending("A person approaches the stand")
    previous_thread = app._reconcile_thread
    previous_started = app._last_reconcile_started_at
    previous_controls = dict(app._runtime_controls)
    try:
        app._reconcile_thread = None
        app._last_reconcile_started_at = 0.0
        app._runtime_controls["reconcile"] = True
        app._start_reconcile(world, {"name": "test"}, people=MagicMock())
        assert world.pending == []
    finally:
        app._reconcile_thread = previous_thread
        app._last_reconcile_started_at = previous_started
        app._runtime_controls.clear()
        app._runtime_controls.update(previous_controls)


def test_start_reconcile_batches_when_thread_alive_or_recent(monkeypatch):
    world = WorldState()
    world.add_pending("A person approaches the stand")
    previous_thread = app._reconcile_thread
    previous_started = app._last_reconcile_started_at
    previous_controls = dict(app._runtime_controls)
    fake_thread = MagicMock()
    fake_thread.is_alive.return_value = True
    try:
        app._reconcile_thread = fake_thread
        app._last_reconcile_started_at = 0.0
        app._runtime_controls["reconcile"] = True
        app._start_reconcile(world, {"name": "test"}, people=MagicMock())
        assert world.pending == ["A person approaches the stand"]

        app._reconcile_thread = None
        app._last_reconcile_started_at = 100.0
        monkeypatch.setattr(app.time, "time", lambda: 100.5)
        app._start_reconcile(world, {"name": "test"}, people=MagicMock())
        assert world.pending == ["A person approaches the stand"]
    finally:
        app._reconcile_thread = previous_thread
        app._last_reconcile_started_at = previous_started
        app._runtime_controls.clear()
        app._runtime_controls.update(previous_controls)


def test_deliver_beat_attaches_actual_planner_prompt_blocks():
    class DummySession:
        def __init__(self):
            self.added = []

        def add_assistant(self, content):
            self.added.append(content)

    class DummyBeat:
        line = "You look like you're posing for a photo. What time is it?"

    pushed = []
    previous_push = app._push
    previous_meta = app._active_turn_meta
    previous_prompt_blocks = app._prompt_trace_blocks_for_labels
    try:
        app._push = lambda event_type, data: pushed.append((event_type, data))
        app._prompt_trace_blocks_for_labels = lambda labels, **kwargs: [{
            "label": "next_beat",
            "messages": [{"role": "user", "content": "Planner saw the recent conversation."}],
            "output": DummyBeat.line,
        }]
        app._active_turn_meta = {"turn_kind": "reaction", "trigger_reason": "visual_stage_exit"}
        response = app.deliver_beat(DummySession(), DummyBeat(), "Greg")
    finally:
        app._push = previous_push
        app._prompt_trace_blocks_for_labels = previous_prompt_blocks
        app._active_turn_meta = previous_meta

    assert response == DummyBeat.line
    response_done = next(data for event_type, data in pushed if event_type == "response_done")
    assert response_done["prompt_blocks"]
    assert response_done["prompt_blocks"][0]["label"] == "next_beat"
    assert response_done["prompt_blocks"][0]["messages"][-1]["content"] == "Planner saw the recent conversation."


def test_stream_response_without_message_uses_session_history(monkeypatch):
    class DummySession:
        def __init__(self):
            self.respond_called = False

        def respond(self):
            self.respond_called = True
            yield "Fresh response"

        def send(self, _message):
            raise AssertionError("send() should not be used for autonomous responses")

        def upsert_system(self, *_args, **_kwargs):
            return None

        def remove_tagged_system(self, *_args, **_kwargs):
            return None

        def get_history(self):
            return [{"role": "user", "content": "What time is it?"}]

    session = DummySession()
    pushed = []
    previous_push = app._push
    try:
        app._push = lambda event_type, data: pushed.append((event_type, data))
        response = app.stream_response(session, "Greg", None)
    finally:
        app._push = previous_push

    assert response == "Fresh response"
    assert session.respond_called is True
    response_done = next(data for event_type, data in pushed if event_type == "response_done")
    assert response_done["turn_outcome"] == "replied"


def test_stream_response_scrubs_pre_audio_cancelled_partial_from_history():
    voice = MagicMock()
    voice._cancelled = threading.Event()
    voice._is_speaking = False
    voice._barged_in = False
    voice._tts = MagicMock()
    voice._speaker = MagicMock()
    voice._speaker._audio_started = threading.Event()
    voice._speaker.wait_for_audio.return_value = False
    voice._speaker.is_done.return_value = True
    voice.audio_started = False
    voice.speaker_playback_ratio = 0.0
    voice.begin_assistant_turn = MagicMock()
    voice.finish_assistant_turn = MagicMock()
    voice.note_assistant_text = MagicMock()
    voice.cancel_latency_filler = MagicMock()
    voice.start_latency_filler = MagicMock()

    class DummySession:
        def __init__(self, live_voice):
            self.live_voice = live_voice
            self.messages = []

        def send(self, _message):
            def gen():
                try:
                    self.live_voice._cancelled.set()
                    yield "I"
                finally:
                    self.messages.append({"role": "assistant", "content": "I"})
            return gen()

        def respond(self):
            raise AssertionError("respond() should not be used for direct user turns")

        def replace_last_assistant(self, content):
            for i in range(len(self.messages) - 1, -1, -1):
                if self.messages[i]["role"] != "assistant":
                    continue
                if content:
                    self.messages[i]["content"] = content
                else:
                    self.messages.pop(i)
                return

        def upsert_system(self, *_args, **_kwargs):
            return None

        def remove_tagged_system(self, *_args, **_kwargs):
            return None

        def get_history(self):
            return [{"role": "user", "content": "fair"}]

    session = DummySession(voice)
    pushed = []
    previous_push = app._push
    try:
        app._push = lambda event_type, data: pushed.append((event_type, data))
        response = app.stream_response(session, "Greg", "fair", voice_io=voice)
    finally:
        app._push = previous_push

    assert response == ""
    assert app.stream_response._cancelled_reply is True
    assert session.messages == []
    assert not any(event_type == "response_chunk" for event_type, _ in pushed)
    response_done = next(data for event_type, data in pushed if event_type == "response_done")
    assert response_done["turn_outcome"] == "cancelled"
    assert response_done["full_text"] == ""
