#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from character_eng.config import load_config
from character_eng.livekit_auth import build_room_name, issue_participant_token, livekit_status_payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue a LiveKit participant token from local config/env")
    parser.add_argument("--room", help="Existing room name to join")
    parser.add_argument("--purpose", default="remote-hot", help="Room purpose label when auto-generating the room name")
    parser.add_argument("--character", default="greg", help="Character name for generated room names")
    parser.add_argument("--identity", required=True, help="Participant identity")
    parser.add_argument("--name", default="", help="Participant display name")
    parser.add_argument("--metadata-json", default="", help="Optional participant metadata as JSON")
    args = parser.parse_args()

    cfg = load_config()
    status = livekit_status_payload(cfg.livekit)
    if not status["configured"]:
        print(json.dumps({"ok": False, "error": "LiveKit is not configured"}))
        return 2

    metadata = None
    if args.metadata_json.strip():
        metadata = json.loads(args.metadata_json)

    room_name = args.room or build_room_name(cfg.livekit, purpose=args.purpose, character=args.character)
    payload = issue_participant_token(
        cfg.livekit,
        room_name=room_name,
        identity=args.identity,
        participant_name=args.name or args.identity,
        metadata=metadata,
    )
    print(json.dumps({"ok": True, **payload.asdict()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
