from character_eng.robot_preview import compile_robot_preview
from character_eng.robot_preview_motion import trajectory_for_preview


def test_compile_robot_preview_defaults_to_idle_keepalive():
    payload = compile_robot_preview({
        "face": {"expression": "neutral", "gaze": "scene", "gaze_type": "hold"},
        "wave": {"active": False, "seconds_left": 0.0},
        "fistbump": {"state": "idle", "active": False},
        "headline": "Idle",
        "updated_at": 12.0,
    })

    assert payload["ok"] is True
    assert payload["body"]["mode"] == "idle_keepalive"
    assert payload["body"]["active_clip"]["key"] == "idle.keepalive"
    assert payload["keepalive"]["state"] == "active"
    assert payload["rust_eyes"]["enabled"] is False
    assert payload["dispatch"]["mode"] == "preview_only"


def test_compile_robot_preview_prioritizes_interaction_over_wave():
    payload = compile_robot_preview({
        "face": {"expression": "curious", "gaze": "left_table", "gaze_type": "glance"},
        "wave": {"active": True, "seconds_left": 1.1},
        "fistbump": {
            "state": "offered",
            "active": True,
            "session_id": "fb-123",
            "can_bump": True,
            "seconds_left": 6.0,
            "note": "Waiting for contact.",
        },
        "rust_eyes": {"enabled": True, "state": "ok", "expression_key": "curious"},
    })

    assert payload["body"]["mode"] == "fistbump_offer"
    assert payload["body"]["active_clip"]["key"] == "fistbump.offer"
    assert payload["interaction"]["session_id"] == "fb-123"
    assert payload["interaction"]["can_bump"] is True
    assert payload["rust_eyes"]["expression_key"] == "curious"


def test_compile_robot_preview_marks_wave_clip_when_active():
    payload = compile_robot_preview({
        "face": {"expression": "happy", "gaze": "scene", "gaze_type": "hold"},
        "wave": {"active": True, "seconds_left": 1.4},
        "fistbump": {"state": "idle", "active": False},
    })

    assert payload["body"]["mode"] == "wave"
    assert payload["body"]["active_clip"]["key"] == "gesture.wave"
    assert payload["keepalive"]["state"] == "suppressed"


def test_trajectory_for_preview_returns_idle_loop():
    preview = compile_robot_preview({
        "face": {"expression": "neutral", "gaze": "scene", "gaze_type": "hold"},
        "wave": {"active": False, "seconds_left": 0.0},
        "fistbump": {"state": "idle", "active": False},
    })

    trajectory = trajectory_for_preview(preview)

    assert trajectory["ok"] is True
    assert trajectory["robot"] == "booster_k1"
    assert trajectory["loop"] is True
    assert trajectory["clip_key"] == "idle.keepalive"
    assert len(trajectory["frames"]) > 10
    assert len(trajectory["frames"][0]) == len(trajectory["joint_names"])


def test_trajectory_for_preview_returns_wave_clip():
    preview = compile_robot_preview({
        "face": {"expression": "happy", "gaze": "scene", "gaze_type": "hold"},
        "wave": {"active": True, "seconds_left": 1.4},
        "fistbump": {"state": "idle", "active": False},
    })

    trajectory = trajectory_for_preview(preview)

    assert trajectory["loop"] is False
    assert trajectory["clip_key"] == "gesture.wave"
    assert trajectory["summary"]
    assert len(trajectory["frames"]) > 20
