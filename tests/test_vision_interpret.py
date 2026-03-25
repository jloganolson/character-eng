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
