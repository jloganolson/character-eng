from __future__ import annotations

import json
import secrets
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from livekit.api import AccessToken, VideoGrants

from character_eng.config import LiveKitConfig


@dataclass(frozen=True)
class LiveKitTokenPayload:
    server_url: str
    room_name: str
    identity: str
    participant_name: str
    token: str
    metadata: str
    ttl_seconds: int

    def asdict(self) -> dict[str, Any]:
        return {
            "serverUrl": self.server_url,
            "roomName": self.room_name,
            "identity": self.identity,
            "participantName": self.participant_name,
            "token": self.token,
            "metadata": self.metadata,
            "ttlSeconds": self.ttl_seconds,
        }


def livekit_status_payload(cfg: LiveKitConfig) -> dict[str, Any]:
    url = str(cfg.url or "").strip()
    api_key = str(cfg.api_key or "").strip()
    api_secret = str(cfg.api_secret or "").strip()
    enabled = bool(cfg.enabled and url and api_key and api_secret)
    return {
        "enabled": enabled,
        "configured": bool(url and api_key and api_secret),
        "serverUrl": url,
        "roomPrefix": str(cfg.room_prefix or "").strip(),
        "apiKeyPresent": bool(api_key),
        "apiSecretPresent": bool(api_secret),
    }


def build_room_name(cfg: LiveKitConfig, *, purpose: str, character: str = "") -> str:
    prefix = (cfg.room_prefix or "character-eng").strip() or "character-eng"
    parts = [prefix, purpose.strip() or "session"]
    if character.strip():
        parts.append(character.strip().replace(" ", "-"))
    parts.append(secrets.token_hex(4))
    return "-".join(parts)


def issue_participant_token(
    cfg: LiveKitConfig,
    *,
    room_name: str,
    identity: str,
    participant_name: str = "",
    metadata: dict[str, Any] | str | None = None,
    can_publish: bool = True,
    can_subscribe: bool = True,
    can_publish_data: bool = True,
    ttl: timedelta = timedelta(hours=8),
) -> LiveKitTokenPayload:
    status = livekit_status_payload(cfg)
    if not status["configured"]:
        raise ValueError("LiveKit is not configured")
    room_name = room_name.strip()
    identity = identity.strip()
    if not room_name:
        raise ValueError("room_name is required")
    if not identity:
        raise ValueError("identity is required")

    metadata_text = ""
    if isinstance(metadata, dict):
        metadata_text = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    elif metadata is not None:
        metadata_text = str(metadata)

    token = (
        AccessToken(api_key=cfg.api_key, api_secret=cfg.api_secret)
        .with_identity(identity)
        .with_name(participant_name or identity)
        .with_metadata(metadata_text)
        .with_ttl(ttl)
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=can_publish,
                can_subscribe=can_subscribe,
                can_publish_data=can_publish_data,
            )
        )
        .to_jwt()
    )
    return LiveKitTokenPayload(
        server_url=cfg.url,
        room_name=room_name,
        identity=identity,
        participant_name=participant_name or identity,
        token=token,
        metadata=metadata_text,
        ttl_seconds=int(ttl.total_seconds()),
    )
