from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any


ASSET_ROOT = Path(__file__).parent / "dashboard" / "assets"
MODEL_ROOT = ASSET_ROOT / "booster_k1"
MODEL_XML_PATH = MODEL_ROOT / "K1_serial.xml"
DEFAULT_POSE_PATH = MODEL_ROOT / "default-pose.json"
WAVE_CLIP_PATH = ASSET_ROOT / "motions" / "booster-wave.json"
ROOT_POS = [0.0, 0.0, 0.457]
ROOT_QUAT_WXYZ = [1.0, 0.0, 0.0, 0.0]
FPS = 60

MUJOCO_JOINT_ORDER = [
    "AAHead_yaw",
    "Head_pitch",
    "ALeft_Shoulder_Pitch",
    "Left_Shoulder_Roll",
    "Left_Elbow_Pitch",
    "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]

SDK_TO_MUJOCO = {
    "Head Yaw": "AAHead_yaw",
    "Head Pitch": "Head_pitch",
    "L Shoulder Pitch": "ALeft_Shoulder_Pitch",
    "L Shoulder Roll": "Left_Shoulder_Roll",
    "L Shoulder Yaw": "Left_Elbow_Pitch",
    "L Elbow": "Left_Elbow_Yaw",
    "R Shoulder Pitch": "ARight_Shoulder_Pitch",
    "R Shoulder Roll": "Right_Shoulder_Roll",
    "R Shoulder Yaw": "Right_Elbow_Pitch",
    "R Elbow": "Right_Elbow_Yaw",
    "L Hip Pitch": "Left_Hip_Pitch",
    "L Hip Roll": "Left_Hip_Roll",
    "L Hip Yaw": "Left_Hip_Yaw",
    "L Knee": "Left_Knee_Pitch",
    "L Ankle Up": "Left_Ankle_Pitch",
    "L Ankle Down": "Left_Ankle_Roll",
    "R Hip Pitch": "Right_Hip_Pitch",
    "R Hip Roll": "Right_Hip_Roll",
    "R Hip Yaw": "Right_Hip_Yaw",
    "R Knee": "Right_Knee_Pitch",
    "R Ankle Up": "Right_Ankle_Pitch",
    "R Ankle Down": "Right_Ankle_Roll",
}


def _lerp(a: float, b: float, t: float) -> float:
    return a + ((b - a) * t)


def _ease_in_out(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


@lru_cache(maxsize=1)
def _base_pose() -> dict[str, float]:
    payload = json.loads(DEFAULT_POSE_PATH.read_text())
    sdk_pose = payload.get("background_joint_pos") or payload.get("non_animated_joint_pos") or {}
    result = {joint: 0.0 for joint in MUJOCO_JOINT_ORDER}
    for sdk_name, value in sdk_pose.items():
        mujoco_name = SDK_TO_MUJOCO.get(str(sdk_name))
        if mujoco_name:
            result[mujoco_name] = float(value)
    return result


@lru_cache(maxsize=1)
def _wave_frames() -> list[list[float]]:
    payload = json.loads(WAVE_CLIP_PATH.read_text())
    frames: list[list[float]] = []
    base_pose = _base_pose()
    for frame in payload.get("frames", []):
        pose = dict(base_pose)
        for sdk_name, value in dict(frame.get("dof_pos") or {}).items():
            mujoco_name = SDK_TO_MUJOCO.get(str(sdk_name))
            if mujoco_name:
                pose[mujoco_name] = float(value)
        frames.append([pose[joint] for joint in MUJOCO_JOINT_ORDER])
    return frames


def _idle_keepalive_frames() -> list[list[float]]:
    base_pose = _base_pose()
    duration_s = 4.0
    frame_count = int(duration_s * FPS)
    frames: list[list[float]] = []
    for idx in range(frame_count):
        t = idx / FPS
        pose = dict(base_pose)
        pose["AAHead_yaw"] += 0.08 * math.sin((2.0 * math.pi * t) / duration_s)
        pose["Head_pitch"] += 0.04 * math.sin((2.0 * math.pi * t) / (duration_s * 0.5) + 0.8)
        pose["ALeft_Shoulder_Pitch"] += 0.03 * math.sin((2.0 * math.pi * t) / 2.4 + 0.2)
        pose["ARight_Shoulder_Pitch"] += 0.03 * math.sin((2.0 * math.pi * t) / 2.4 + 1.8)
        pose["Left_Hip_Roll"] += 0.025 * math.sin((2.0 * math.pi * t) / 1.8)
        pose["Right_Hip_Roll"] -= 0.025 * math.sin((2.0 * math.pi * t) / 1.8)
        pose["Left_Ankle_Roll"] += 0.03 * math.sin((2.0 * math.pi * t) / 1.8)
        pose["Right_Ankle_Roll"] -= 0.03 * math.sin((2.0 * math.pi * t) / 1.8)
        frames.append([pose[joint] for joint in MUJOCO_JOINT_ORDER])
    return frames


def _fistbump_offer_frames() -> list[list[float]]:
    base_pose = _base_pose()
    target_pose = dict(base_pose)
    target_pose["AAHead_yaw"] = -0.12
    target_pose["Head_pitch"] += 0.05
    target_pose["ARight_Shoulder_Pitch"] = -0.55
    target_pose["Right_Shoulder_Roll"] = 0.62
    target_pose["Right_Elbow_Pitch"] = 0.92
    target_pose["Right_Elbow_Yaw"] = 0.42
    target_pose["ALeft_Shoulder_Pitch"] += 0.08
    target_pose["Left_Shoulder_Roll"] -= 0.04

    frames: list[list[float]] = []
    duration_s = 0.9
    frame_count = max(2, int(duration_s * FPS))
    for idx in range(frame_count):
        t = _ease_in_out(idx / (frame_count - 1))
        pose = {
            joint: _lerp(base_pose[joint], target_pose[joint], t)
            for joint in MUJOCO_JOINT_ORDER
        }
        frames.append([pose[joint] for joint in MUJOCO_JOINT_ORDER])
    return frames


def _fistbump_retract_frames() -> list[list[float]]:
    offer_frames = _fistbump_offer_frames()
    return list(reversed(offer_frames))


def trajectory_for_preview(preview_state: dict[str, Any]) -> dict[str, Any]:
    body = dict(preview_state.get("body") or {})
    clip = dict(body.get("active_clip") or {})
    clip_key = str(clip.get("key") or "idle.keepalive")
    body_mode = str(body.get("mode") or "idle_keepalive")

    if clip_key == "gesture.wave":
        frames = _wave_frames()
        loop = False
        hold_last = False
        summary = "Duplicated authored wave clip rendered from local assets."
    elif body_mode == "fistbump_offer":
        frames = _fistbump_offer_frames()
        loop = False
        hold_last = True
        summary = "Local authored fist bump offer pose sequence."
    elif body_mode in {"fistbump_contact", "fistbump_retract"}:
        frames = _fistbump_retract_frames()
        loop = False
        hold_last = False
        summary = "Local retract sequence returning to the base pose."
    else:
        frames = _idle_keepalive_frames()
        loop = True
        hold_last = False
        summary = "Procedural idle keepalive generated in character-eng."

    return {
        "ok": True,
        "robot": "booster_k1",
        "clip_key": clip_key,
        "body_mode": body_mode,
        "fps": FPS,
        "duration_s": len(frames) / FPS,
        "loop": loop,
        "hold_last_frame": hold_last,
        "root_pos": ROOT_POS,
        "root_quat_wxyz": ROOT_QUAT_WXYZ,
        "joint_names": MUJOCO_JOINT_ORDER,
        "frames": frames,
        "summary": summary,
    }
