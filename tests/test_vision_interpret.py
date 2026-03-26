import json

from character_eng.person import PeopleState
from character_eng.vision.interpret import vision_state_update_call
from character_eng.world import WorldState


def test_vision_state_update_call_parses_world_and_person_updates(monkeypatch):
    class Message:
        content = json.dumps({
            "thought": "Static appearance should live on the person.",
            "summary": "Visitor is wearing a blue jacket",
            "remove_facts": ["f1"],
            "add_facts": ["A wall clock is visible"],
            "events": ["The visitor is checking a phone"],
            "person_updates": [
                {
                    "person_id": "p1",
                    "add_facts": ["Wearing a blue jacket"],
                    "fact_scope": "static",
                    "set_presence": "present",
                }
            ],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    world = WorldState()
    world.add_fact("Old fact")
    people = PeopleState()
    people.add_person(name="Visitor", presence="present")
    result = vision_state_update_call(
        world=world,
        people=people,
        task_answers=[
            {
                "task_id": "person_description_static",
                "label": "Person Description Static",
                "question": "Describe the nearest person.",
                "answer": "The visitor is wearing a blue jacket.",
                "interpret_as": "person_description_static",
                "target": "nearest_person",
                "target_person_id": "p1",
            }
        ],
        model_config={},
    )

    assert result.thought == "Static appearance should live on the person."
    assert result.summary == "Visitor is wearing a blue jacket"
    assert result.update.remove_facts == ["f1"]
    assert result.update.add_facts == ["A wall clock is visible"]
    assert result.update.events == ["The visitor is checking a phone"]
    assert result.update.person_updates[0].person_id == "p1"
    assert result.update.person_updates[0].add_facts == ["Wearing a blue jacket"]
    assert result.update.person_updates[0].fact_scope == "static"


def test_vision_state_update_call_dedupes_existing_person_facts_from_world_state(monkeypatch):
    class Message:
        content = json.dumps({
            "thought": "",
            "summary": "Updated person description for Person 1.",
            "remove_facts": [],
            "add_facts": [
                "Person 1 is wearing glasses.",
                "The person remains stationary.",
                "The room has white walls.",
            ],
            "events": [],
            "person_updates": [
                {
                    "person_id": "p1",
                    "add_facts": ["Person 1 is wearing glasses."],
                    "fact_scope": "static",
                    "set_name": None,
                }
            ],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    result = vision_state_update_call(
        world=WorldState(),
        people=PeopleState(),
        task_answers=[
            {
                "task_id": "person_description_dynamic_person_1",
                "label": "Person Description Dynamic [Person 1]",
                "question": "Describe Person 1.",
                "answer": "Person 1 is wearing glasses and standing still.",
                "interpret_as": "person_description_dynamic",
                "target": "person_identity",
                "target_person_id": "p1",
                "target_identity": "Person 1",
            }
        ],
        model_config={},
    )

    assert result.update.add_facts == ["The room has white walls."]
    assert len(result.update.person_updates) == 1
    assert result.update.person_updates[0].add_facts == [
        "Person 1 is wearing glasses.",
        "The person remains stationary.",
    ]
    assert result.update.person_updates[0].set_name is None


def test_vision_state_update_call_treats_null_set_name_as_missing(monkeypatch):
    class Message:
        content = json.dumps({
            "thought": "",
            "summary": "",
            "remove_facts": [],
            "add_facts": [],
            "events": [],
            "person_updates": [
                {
                    "person_id": "p1",
                    "add_facts": ["Looking left"],
                    "fact_scope": "ephemeral",
                    "set_name": None,
                }
            ],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    result = vision_state_update_call(
        world=WorldState(),
        people=PeopleState(),
        task_answers=[
            {
                "task_id": "person_description_dynamic_person_1",
                "label": "Person Description Dynamic [Person 1]",
                "question": "Describe Person 1.",
                "answer": "Looking left.",
                "interpret_as": "person_description_dynamic",
                "target": "person_identity",
                "target_person_id": "p1",
                "target_identity": "Person 1",
            }
        ],
        model_config={},
    )

    assert result.update.person_updates[0].set_name is None


def test_vision_state_update_call_strips_unsupported_set_name(monkeypatch):
    class Message:
        content = json.dumps({
            "thought": "",
            "summary": "Updated person description.",
            "remove_facts": [],
            "add_facts": [],
            "events": [],
            "person_updates": [
                {
                    "person_id": "p1",
                    "add_facts": ["The person is wearing glasses."],
                    "fact_scope": "static",
                    "set_name": "Greg",
                }
            ],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    people = PeopleState()
    people.add_person(presence="present")
    result = vision_state_update_call(
        world=WorldState(),
        people=people,
        task_answers=[
            {
                "task_id": "person_description_static",
                "label": "Person Description Static",
                "question": "Describe the nearest person.",
                "answer": "The person is wearing glasses.",
                "interpret_as": "person_description_static",
                "target": "nearest_person",
                "target_person_id": "p1",
                "target_identity": "Person 2",
            }
        ],
        model_config={},
    )

    assert len(result.update.person_updates) == 1
    assert result.update.person_updates[0].set_name is None


def test_vision_state_update_call_ignores_null_optional_person_fields(monkeypatch):
    class Message:
        content = json.dumps({
            "person_updates": [
                {
                    "person_id": "p1",
                    "add_facts": ["Wearing a blue jacket"],
                    "set_name": None,
                    "set_presence": None,
                    "fact_scope": None,
                }
            ],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    result = vision_state_update_call(
        world=WorldState(),
        people=PeopleState(),
        task_answers=[
            {
                "task_id": "person_description_static",
                "label": "Person Description Static",
                "question": "Describe the nearest person.",
                "answer": "The visitor is wearing a blue jacket.",
                "interpret_as": "person_description_static",
                "target": "nearest_person",
                "target_person_id": "p1",
            }
        ],
        model_config={},
    )

    update = result.update.person_updates[0]
    assert update.set_name is None
    assert update.set_presence is None
    assert update.fact_scope is None


def test_vision_state_update_call_moves_person_facts_out_of_world_state(monkeypatch):
    class Message:
        content = json.dumps({
            "add_facts": [
                "The person is wearing a patterned shirt",
                "A wall clock is visible",
            ],
            "person_updates": [],
        })

    class Choice:
        message = Message()

    class Response:
        choices = [Choice()]

    monkeypatch.setattr("character_eng.world._llm_call", lambda *args, **kwargs: Response())

    result = vision_state_update_call(
        world=WorldState(),
        people=PeopleState(),
        task_answers=[
            {
                "task_id": "world_state",
                "label": "World State",
                "question": "Describe the visible room.",
                "answer": "The person is wearing a patterned shirt. A wall clock is visible.",
                "interpret_as": "world_state",
                "target": "scene",
                "target_person_id": "p1",
            }
        ],
        model_config={},
    )

    assert result.update.add_facts == ["A wall clock is visible"]
    assert len(result.update.person_updates) == 1
    assert result.update.person_updates[0].person_id == "p1"
    assert result.update.person_updates[0].add_facts == ["The person is wearing a patterned shirt"]
