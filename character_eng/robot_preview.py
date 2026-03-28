from __future__ import annotations

from typing import Any


def _face_payload(face: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(face or {})
    payload["gaze"] = str(payload.get("gaze") or "scene")
    payload["gaze_type"] = str(payload.get("gaze_type") or "hold")
    payload["expression"] = str(payload.get("expression") or "neutral")
    payload["source"] = str(payload.get("source") or "runtime")
    return payload


def _rust_eyes_payload(rust_eyes: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(rust_eyes or {})
    payload["enabled"] = bool(payload.get("enabled"))
    payload["state"] = str(payload.get("state") or ("disabled" if not payload["enabled"] else "configured"))
    payload["message"] = str(payload.get("message") or "")
    payload["expression_key"] = str(payload.get("expression_key") or "")
    return payload


def _body_program(
    *,
    wave: dict[str, Any],
    fistbump: dict[str, Any],
) -> dict[str, Any]:
    fist_state = str(fistbump.get("state") or "idle")
    wave_active = bool(wave.get("active"))

    if fist_state == "offered":
        clip = {
            "key": "fistbump.offer",
            "label": "Fist bump offer",
            "playback": "hold_until_event",
            "loop": False,
            "seconds_left": float(fistbump.get("seconds_left") or 0.0),
        }
        return {
            "mode": "fistbump_offer",
            "pose_family": "interaction",
            "active_clip": clip,
            "summary": "Arm extension is active and waiting for contact or timeout.",
        }

    if fist_state == "contacted":
        clip = {
            "key": "fistbump.contact_retract",
            "label": "Fist bump retract",
            "playback": "result_hold",
            "loop": False,
            "seconds_left": 0.0,
        }
        return {
            "mode": "fistbump_contact",
            "pose_family": "interaction",
            "active_clip": clip,
            "summary": "Contact has landed and the arm is in the retract/settle phase.",
        }

    if fist_state in {"timed_out", "cancelled"}:
        clip = {
            "key": "fistbump.retract",
            "label": "Fist bump retract",
            "playback": "oneshot",
            "loop": False,
            "seconds_left": 0.0,
        }
        return {
            "mode": "fistbump_retract",
            "pose_family": "interaction",
            "active_clip": clip,
            "summary": "The interaction ended without a held offer, so the body is retracting.",
        }

    if wave_active:
        clip = {
            "key": "gesture.wave",
            "label": "Wave",
            "playback": "oneshot",
            "loop": False,
            "seconds_left": float(wave.get("seconds_left") or 0.0),
        }
        return {
            "mode": "wave",
            "pose_family": "gesture",
            "active_clip": clip,
            "summary": "A triggered greeting clip is currently active.",
        }

    clip = {
        "key": "idle.keepalive",
        "label": "Idle keepalive",
        "playback": "loop",
        "loop": True,
        "seconds_left": 0.0,
    }
    return {
        "mode": "idle_keepalive",
        "pose_family": "keepalive",
        "active_clip": clip,
        "summary": "No triggered gesture is active; the body should run a procedural idle program.",
    }


def _keepalive_program(body_mode: str) -> dict[str, Any]:
    if body_mode == "idle_keepalive":
        return {
            "preset": "idle_attentive_v1",
            "state": "active",
            "procedural": True,
            "summary": "Low-amplitude torso, head, and leg motion for a conversational idle loop.",
        }
    return {
        "preset": "idle_attentive_v1",
        "state": "suppressed",
        "procedural": True,
        "summary": "Keepalive is paused or blended down while a higher-priority gesture runs.",
    }


def compile_robot_preview(robot_state: dict[str, Any] | None) -> dict[str, Any]:
    state = dict(robot_state or {})
    face = _face_payload(state.get("face"))
    wave = dict(state.get("wave") or {})
    fistbump = dict(state.get("fistbump") or {})
    rust_eyes = _rust_eyes_payload(state.get("rust_eyes"))

    body = _body_program(wave=wave, fistbump=fistbump)
    keepalive = _keepalive_program(body["mode"])
    interaction_state = str(fistbump.get("state") or "idle")

    return {
        "ok": True,
        "headline": str(state.get("headline") or "Idle"),
        "updated_at": float(state.get("updated_at") or 0.0),
        "source": str(state.get("source") or face.get("source") or "runtime"),
        "face": face,
        "wave": wave,
        "fistbump": fistbump,
        "rust_eyes": rust_eyes,
        "body": body,
        "keepalive": keepalive,
        "interaction": {
            "state": interaction_state,
            "active": bool(fistbump.get("active")),
            "session_id": str(fistbump.get("session_id") or ""),
            "can_bump": bool(fistbump.get("can_bump")),
            "seconds_left": float(fistbump.get("seconds_left") or 0.0),
            "note": str(fistbump.get("note") or ""),
        },
        "dispatch": {
            "authority": "character_eng",
            "mode": "preview_only",
            "preview_transport": "browser_mujoco_pending",
            "hardware_transport": "booster_sdk_pending",
            "ready_for_hardware": False,
            "summary": "Preview routes are live in this repo. Hardware dispatch is not wired yet.",
        },
        "authoring": {
            "surface": "robot_preview",
            "supports_idle_authoring": True,
            "supports_triggered_clips": True,
            "next_targets": ["idle_keepalive", "gesture.wave", "fistbump.offer"],
        },
        "robot_sim": state,
    }
