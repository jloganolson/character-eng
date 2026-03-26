from character_eng.name_memory import (
    apply_transcript_name_to_people,
    extract_self_identified_name,
    has_explicit_name_evidence,
)
from character_eng.person import PeopleState


def test_extract_self_identified_name_from_transcript():
    assert extract_self_identified_name("that's a lot man my name is logan") == "Logan"
    assert extract_self_identified_name("i'm alex") == "Alex"
    assert extract_self_identified_name("i am tired") is None


def test_has_explicit_name_evidence_requires_name_claim():
    assert has_explicit_name_evidence("Logan", ["my name is logan"]) is True
    assert has_explicit_name_evidence("Greg", ["The person is holding a mug."]) is False


def test_apply_transcript_name_to_people_updates_single_present_person():
    people = PeopleState()
    people.add_person(name="Greg", presence="present")

    person_id, captured_name = apply_transcript_name_to_people(people, "that's a lot man my name is logan")

    assert person_id == "p1"
    assert captured_name == "Logan"
    assert people.people["p1"].name == "Logan"
    assert "Logan" in people.people["p1"].aliases
