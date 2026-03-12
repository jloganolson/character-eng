import json

from character_eng.history import (
    HistoryService,
    PlaybackRunner,
    load_checkpoint,
    resolve_session_path,
    restore_runtime_state,
)
from character_eng.person import PeopleState
from character_eng.world import Goals, Script, WorldState


def _session_snapshot():
    return {
        "system_prompt": "system",
        "messages": [{"role": "system", "content": "system"}],
        "tagged_system_indices": {},
    }


def test_history_service_records_session_checkpoint_and_restore(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess1", character="greg", model="test")

    world = WorldState(static=["robot"], dynamic={"f1": "cup on table"}, events=["boot"])
    people = PeopleState()
    person = people.add_person(name="Visitor", presence="present")
    person.add_fact("holding phone")
    script = Script()
    goals = Goals(long_term="be helpful")

    archive.record_event("session_start", {"session_id": "sess1"})
    checkpoint_path = archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=world,
        people=people,
        scenario=None,
        script=script,
        goals=goals,
        log_entries=[{"type": "send", "input": "hi", "response": "hello"}],
        context_version=2,
        had_user_input=True,
    )
    assert checkpoint_path.exists()

    payload = load_checkpoint(archive.path)
    restored = restore_runtime_state(payload)
    assert restored["world"].dynamic["f1"] == "cup on table"
    assert restored["people"].people["p1"].facts["p1f1"] == "holding phone"
    assert restored["goals"].long_term == "be helpful"
    assert restored["log_entries"][0]["response"] == "hello"


def test_annotations_auto_promote_and_catalog(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    service.start_session(session_id="sess2", character="greg", model="test")
    service.finalize_current()

    path = service.save_annotation(
        {
            "session_id": "sess2",
            "note": "Greg interrupted the user.",
            "tags": ["interruption"],
        },
        session_id="sess2",
    )
    assert path.exists()
    pinned = resolve_session_path(tmp_path, "sess2")
    assert pinned == tmp_path / "sessions" / "sess2"
    assert (tmp_path / "pinned" / "sess2").exists()
    catalog = tmp_path / "catalog" / "annotations.jsonl"
    assert catalog.exists()
    assert "Greg interrupted the user." in catalog.read_text()


def test_capture_moment_writes_audio_and_video_context(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess3", character="greg", model="test")
    archive.record_event("session_start", {"session_id": "sess3"})
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_assistant_audio(b"\x00\x00" * 24000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.5, source="vision")
    service.finalize_current()

    moment_dir = service.capture_moment(
        {
            "session_id": "sess3",
            "title": "phone moment",
            "event_time_s": 0.5,
            "bundle": {"event": {"type": "vision_pass"}},
            "window_before_s": 0.25,
            "window_after_s": 0.25,
        }
    )
    assert (moment_dir / "manifest.json").exists()
    assert (moment_dir / "checkpoint.json").exists()
    assert (moment_dir / "media" / "audio" / "user_input_clip.wav").exists()
    assert (moment_dir / "media" / "audio" / "assistant_output_clip.wav").exists()
    assert (moment_dir / "media" / "video" / "frame_000000.jpg").exists()


def test_prune_unpromoted_sessions_leaves_pinned(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    service.start_session(session_id="keep", character="greg", model="test")
    service.finalize_current()
    service.promote(session_id="keep", kind="pinned")

    service.start_session(session_id="drop", character="greg", model="test")
    service.finalize_current()

    result = service.prune_unpromoted(older_than_days=0)
    assert str(tmp_path / "sessions" / "drop") in result["removed"]
    assert (tmp_path / "sessions" / "keep").exists()
    assert not (tmp_path / "sessions" / "drop").exists()


def test_prepare_playback_for_session_uses_checkpoint_offset(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess4", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x00\x00" * 16000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.25, source="vision")

    later_checkpoint = archive.capture_checkpoint(
        label="later",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(dynamic={"f1": "phone out"}),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=1,
        had_user_input=True,
    )
    payload = json.loads(later_checkpoint.read_text())
    frame_ts = payload["timestamp"] + 0.5
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=frame_ts, source="vision")
    service.finalize_current()

    plan = service.prepare_playback("sess4", 1)
    assert plan.session_id == "sess4"
    assert plan.checkpoint_index == 1
    assert plan.audio_path is not None
    assert plan.audio_start_s >= 0.0
    assert len(plan.video_frames) >= 1
    assert any(round(frame["relative_s"], 3) == 0.5 for frame in plan.video_frames)


def test_prepare_playback_for_moment_uses_clipped_media(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess5", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x00\x00" * 32000)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 1.0, source="vision")
    service.finalize_current()

    moment_dir = service.capture_moment(
        {
            "session_id": "sess5",
            "title": "phone moment",
            "event_time_s": 1.0,
            "window_before_s": 0.5,
            "window_after_s": 0.5,
        }
    )

    plan = service.prepare_playback(str(moment_dir))
    assert plan.source_kind == "moment"
    assert plan.audio_path == moment_dir / "media" / "audio" / "user_input_clip.wav"
    assert plan.audio_start_s == 0.0
    assert len(plan.video_frames) == 1


def test_playback_runner_streams_audio_and_video(tmp_path):
    service = HistoryService(root=tmp_path, free_warning_gib=0.0)
    archive = service.start_session(session_id="sess6", character="greg", model="test")
    archive.capture_checkpoint(
        label="start",
        character="greg",
        session_snapshot=_session_snapshot(),
        world=WorldState(),
        people=PeopleState(),
        scenario=None,
        script=Script(),
        goals=Goals(),
        log_entries=[],
        context_version=0,
        had_user_input=False,
    )
    archive.record_user_audio(b"\x01\x02" * 3200)
    archive.record_video_frame(b"\xff\xd8\xff\xd9", timestamp=archive._started_at + 0.1, source="vision")
    service.finalize_current()

    plan = service.prepare_playback("sess6")
    audio_chunks: list[bytes] = []
    video_chunks: list[dict] = []

    runner = PlaybackRunner(
        plan,
        audio_callback=audio_chunks.append,
        video_callback=lambda jpeg, meta: video_chunks.append({"jpeg": jpeg, **meta}),
        speed=50.0,
        audio_chunk_ms=50,
    )
    runner.start()
    runner.join(timeout=2.0)

    assert audio_chunks
    assert audio_chunks[0]
    assert video_chunks
    assert video_chunks[0]["jpeg"] == b"\xff\xd8\xff\xd9"
