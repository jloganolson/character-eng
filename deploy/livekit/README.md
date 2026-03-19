# LiveKit Remote-Hot Scaffolding

This path is for the future `remote-hot-webrtc` mode only.

It does not replace or modify:
- `./scripts/run_heavy.sh`
- `./scripts/run_hot.sh`
- the existing full-local mic/cam/speaker path

## Why this exists

The current `remote-hot` path works, but it still uses:
- SSH port forwarding
- per-frame JPEG uploads
- ad hoc transport metrics

That was enough to prove the architecture, but it is the wrong long-term transport for low-latency camera/audio.

LiveKit gives us:
- proper realtime audio/video/data transport
- backend-issued participant tokens
- a clean self-hosted path for GCP

## Local dev server

The fastest local-first way to prototype the transport is a disposable LiveKit dev server:

```bash
./deploy/livekit/start_local_dev.sh
cp deploy/livekit.env.example .env.livekit
```

Use these values:

```bash
export CHARACTER_ENG_LIVEKIT_ENABLED=true
export CHARACTER_ENG_LIVEKIT_URL=ws://127.0.0.1:7880
export CHARACTER_ENG_LIVEKIT_API_KEY=devkey
export CHARACTER_ENG_LIVEKIT_API_SECRET=secret
```

Stop it with:

```bash
./deploy/livekit/stop_local_dev.sh
```

## Backend token endpoint

The local app now exposes:
- `GET /livekit/config`
- `POST /livekit/token`

These are served by the same dashboard/bridge HTTP server and are intended to support a future browser/media client without changing the existing full-local path.

Example:

```bash
curl -s http://127.0.0.1:7862/livekit/config | jq
curl -s http://127.0.0.1:7862/livekit/token \
  -H 'Content-Type: application/json' \
  -d '{"purpose":"remote-hot","character":"greg","identity":"logan-dev"}' | jq
```

## Next build step

The next implementation step is to add a separate `remote-hot-webrtc` launcher and frontend/client integration that consumes these endpoints. Until that path is proven, the existing SSH/JPEG `remote-hot` mode remains the fallback.
