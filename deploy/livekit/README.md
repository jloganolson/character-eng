# LiveKit Remote-Hot Transport

This path now powers the local `remote-hot-webrtc` mode.

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
- a remote-heavy GCP mode via `./scripts/run_hot_remote_webrtc.sh`

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

## Backend endpoints

The local app now exposes:
- `GET /livekit/config`
- `POST /livekit/token`

These are served by the same dashboard/bridge HTTP server and are used by the `remote_hot_webrtc` dashboard controls.

Example:

```bash
curl -s http://127.0.0.1:7862/livekit/config | jq
curl -s http://127.0.0.1:7862/livekit/token \
  -H 'Content-Type: application/json' \
  -d '{"purpose":"remote-hot","character":"greg","identity":"logan-dev"}' | jq
```

## Local transport mode

Run:

```bash
./scripts/run_hot_webrtc.sh
```

This does three things:
- verifies the local heavy stack is ready
- starts the LiveKit dev server if needed
- launches the app in `remote_hot_webrtc`

In that mode:
- the header exposes `Arm Audio`, `Mic On`, and `Cam On`
- the Vision panel shows a LiveKit preview of the outgoing camera stream
- the app injects incoming WebRTC video frames into the vision service

The older SSH/JPEG `remote-hot` path remains available as a fallback for GCP-backed remote heavy services.
