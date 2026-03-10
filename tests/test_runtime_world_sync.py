from unittest.mock import MagicMock

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
