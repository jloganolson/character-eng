from __future__ import annotations

import json

import jwt

from character_eng.config import LiveKitConfig
from character_eng.livekit_auth import build_room_name, issue_participant_token, livekit_status_payload


def test_livekit_status_payload_hides_secret_value():
    cfg = LiveKitConfig(
        enabled=True,
        url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="secret",
        room_prefix="ce",
    )
    payload = livekit_status_payload(cfg)
    assert payload["enabled"] is True
    assert payload["configured"] is True
    assert payload["apiKeyPresent"] is True
    assert payload["apiSecretPresent"] is True
    assert "secret" not in json.dumps(payload)


def test_build_room_name_uses_prefix_and_labels():
    cfg = LiveKitConfig(room_prefix="ce")
    room = build_room_name(cfg, purpose="remote-hot", character="greg")
    assert room.startswith("ce-remote-hot-greg-")


def test_issue_participant_token_encodes_join_claims():
    cfg = LiveKitConfig(
        enabled=True,
        url="ws://127.0.0.1:7880",
        api_key="devkey",
        api_secret="secret",
        room_prefix="ce",
    )
    payload = issue_participant_token(
        cfg,
        room_name="ce-room-1",
        identity="logan-dev",
        participant_name="Logan",
        metadata={"mode": "remote-hot-webrtc"},
    )
    claims = jwt.decode(payload.token, "secret", algorithms=["HS256"], audience=None, options={"verify_aud": False})
    assert payload.server_url == "ws://127.0.0.1:7880"
    assert claims["sub"] == "logan-dev"
    assert claims["iss"] == "devkey"
    assert claims["video"]["room"] == "ce-room-1"
    assert claims["video"]["roomJoin"] is True
    assert claims["name"] == "Logan"
    assert json.loads(claims["metadata"]) == {"mode": "remote-hot-webrtc"}
