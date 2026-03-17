from pathlib import Path

from character_eng.qa_vision_snippets import (
    evaluate_vision_state_snippet,
    iter_vision_state_snippets,
    write_vision_state_report,
)
from character_eng.vision.interpret import VisionStateUpdateResult
from character_eng.world import PersonUpdate, WorldUpdate


FIXTURE_ROOT = Path(__file__).resolve().parent / "vision_snippets" / "fixtures"


def test_iter_vision_state_snippets_loads_fixture_and_tags():
    snippets = list(iter_vision_state_snippets(FIXTURE_ROOT, required_tags={"vision-state", "fixture"}))

    assert len(snippets) == 1
    snippet = snippets[0]
    assert snippet.title == "visitor backpack state update"
    assert snippet.capture_mode == "vision_state_turn"
    assert snippet.source_event_type == "vision_state_update"
    assert len(snippet.task_answers) == 2
    assert snippet.expected_update["person_updates"][0]["person_id"] == "p1"


def test_evaluate_vision_state_snippet_matches_expected_update(monkeypatch):
    snippet = next(iter_vision_state_snippets(FIXTURE_ROOT))

    monkeypatch.setattr(
        "character_eng.qa_vision_snippets.vision_state_update_call",
        lambda **kwargs: VisionStateUpdateResult(
            update=WorldUpdate(
                add_facts=["A wall clock is visible above the table."],
                person_updates=[PersonUpdate(person_id="p1", add_facts=["The visitor is wearing a black backpack."])],
            ),
            summary="Visitor now has a backpack",
            thought="Static appearance belongs on the person.",
        ),
    )

    result = evaluate_vision_state_snippet(snippet, model_config={})

    assert result.ok is True
    assert result.summary == "Visitor now has a backpack"
    assert result.missing_world_facts == []
    assert result.extra_world_facts == []
    assert result.unexpected_person_updates == []


def test_evaluate_vision_state_snippet_flags_extra_changes_and_writes_report(monkeypatch, tmp_path):
    snippet = next(iter_vision_state_snippets(FIXTURE_ROOT))

    monkeypatch.setattr(
        "character_eng.qa_vision_snippets.vision_state_update_call",
        lambda **kwargs: VisionStateUpdateResult(
            update=WorldUpdate(
                add_facts=[
                    "A wall clock is visible above the table.",
                    "A second person is standing near the register.",
                ],
                events=["The visitor waved at Greg."],
                person_updates=[
                    PersonUpdate(person_id="p1", add_facts=["The visitor is wearing a black backpack."]),
                    PersonUpdate(person_id="p2", add_facts=["A second person is present."]),
                ],
            ),
            summary="Extra changes appeared",
            thought="The interpreter overreached.",
        ),
    )

    result = evaluate_vision_state_snippet(snippet, model_config={})

    assert result.ok is False
    assert result.extra_world_facts == ["A second person is standing near the register."]
    assert result.extra_events == ["The visitor waved at Greg."]
    assert result.unexpected_person_updates == ["p2"]

    report_path = tmp_path / "vision_state_report.html"
    write_vision_state_report([result], report_path)

    body = report_path.read_text(encoding="utf-8")
    assert "Vision State Snippet Report" in body
    assert "REVIEW" in body
    assert "unexpected person updates" in body
