from unittest.mock import MagicMock

import character_eng.__main__ as app
from character_eng.perception import PerceptionEvent
from character_eng.person import PeopleState
from character_eng.world import RobotActionResult, WorldState


def test_build_robot_action_trigger_includes_people_and_history():
    people = PeopleState()
    pid = people.get_or_create("Logan")
    people.people[pid].presence = "approaching"
    history = [
        {"role": "user", "content": "hi greg"},
        {"role": "assistant", "content": "hello there"},
    ]

    trigger = app._build_robot_action_trigger(
        source="dialogue",
        kind="assistant_reply",
        history=history,
        people=people,
        assistant_reply="hello there",
        summary="Assistant produced a spoken reply.",
    )

    assert trigger.source == "dialogue"
    assert trigger.kind == "assistant_reply"
    assert trigger.latest_user_message == "hi greg"
    assert trigger.assistant_reply == "hello there"
    assert trigger.history == history
    assert "Logan" in trigger.people_text


def test_finalize_robot_action_trigger_pushes_event_and_applies_action():
    trigger = app._build_robot_action_trigger(
        source="dialogue",
        kind="assistant_reply",
        history=[{"role": "user", "content": "hi"}],
        assistant_reply="hello",
        summary="Assistant produced a spoken reply.",
    )
    pushed = []
    applied = {}
    previous_push = app._push
    previous_apply = app._apply_robot_action_result
    try:
        app._push = lambda event_type, data: pushed.append((event_type, data))

        def _fake_apply(result, *, log=None, source=""):
            applied["action"] = result.action
            applied["source"] = source
            return {"action": result.action, "source": source}

        app._apply_robot_action_result = _fake_apply

        payload = app._finalize_robot_action_trigger(
            trigger=trigger,
            result=RobotActionResult(action="wave", reason="greeting"),
            log=[],
            apply_source="llm_dialogue",
        )
    finally:
        app._push = previous_push
        app._apply_robot_action_result = previous_apply

    assert payload == {"action": "wave", "source": "llm_dialogue"}
    assert pushed[0][0] == "robot_action_trigger"
    assert pushed[0][1]["source"] == "dialogue"
    assert pushed[0][1]["kind"] == "assistant_reply"
    assert pushed[0][1]["decision"] == "wave"
    assert pushed[0][1]["reason"] == "greeting"
    assert applied == {"action": "wave", "source": "llm_dialogue"}


def test_consume_vision_updates_uses_shared_robot_action_trigger(monkeypatch):
    class DummySession:
        def get_history(self):
            return [{"role": "user", "content": "hi greg"}]

        def upsert_system(self, *_args, **_kwargs):
            return None

        def remove_tagged_system(self, *_args, **_kwargs):
            return None

    class DummyVisionMgr:
        def __init__(self, events):
            self._events = list(events)
            self.update_context = MagicMock()

        def drain_events(self):
            events = list(self._events)
            self._events.clear()
            return events

    event = PerceptionEvent(
        description="Logan steps into view.",
        source="vision",
        kind="person_presence",
        payload={"identity": "Logan", "presence": "approaching", "signals": ["person_visible"]},
    )
    session = DummySession()
    world = WorldState()
    people = PeopleState()
    vision_mgr = DummyVisionMgr([event])
    captured = {}

    monkeypatch.setattr(app, "_push", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "_push_world_people_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "_start_reconcile", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(app, "match_visual_exit", lambda *_args, **_kwargs: -1)
    monkeypatch.setattr(app, "_should_trigger_visual_turn", lambda *_args, **_kwargs: False)

    def _fake_robot_action_call(*, model_config, trigger, robot_state):
        captured["trigger"] = trigger
        captured["robot_state"] = robot_state
        return RobotActionResult(action="wave", reason="first visual notice")

    def _fake_finalize(*, trigger, result, log=None, apply_source=""):
        captured["finalize_trigger"] = trigger
        captured["finalize_result"] = result
        captured["apply_source"] = apply_source
        return None

    monkeypatch.setattr(app, "robot_action_call", _fake_robot_action_call)
    monkeypatch.setattr(app, "_finalize_robot_action_trigger", _fake_finalize)

    triggered = app._consume_vision_updates(
        session,
        world,
        people,
        vision_mgr,
        scenario=None,
        script=None,
        goals=None,
        model_config={"name": "test"},
        log=[],
    )

    assert triggered is False
    assert captured["trigger"].source == "vision"
    assert captured["trigger"].kind == "perception_batch"
    assert captured["trigger"].latest_user_message == "hi greg"
    assert captured["trigger"].summary == "Vision/perception updates arrived."
    assert any("Logan steps into view." in item for item in captured["trigger"].details)
    assert captured["apply_source"] == "llm_vision"
