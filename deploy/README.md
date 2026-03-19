# Deployment Notes

This repo now has two remote-facing layers:

- `character_eng.session_manager`: spawns one browser-mode runtime per user session
- `character_eng.bridge`: serves the dashboard plus browser mic/camera/audio endpoints for a single runtime

## Container modes

The Docker image uses `deploy/container_entrypoint.sh` and supports:

- `MODE=session-manager`
  - Starts the session manager API on `MANAGER_PORT` (default `7870`)
  - Intended control-plane entrypoint for shared remote use
- `MODE=app`
  - Starts a single browser-mode runtime directly
- `MODE=heavy`
  - Starts the heavy local GPU services (`run_heavy.sh`)
- `MODE=full`
  - Starts heavy services, then the session manager

For fast infrastructure validation, there is also a separate lightweight image path:

- `Dockerfile.manager-lite`
  - Runs `character_eng.session_manager` directly from source
  - Skips `services/vision`, `vllm`, and the full app dependency install
  - Uses a stub per-session runtime to validate RunPod placement, public networking, SSH, manager API, tokenized URLs, and proxying

## Useful env vars

- `PUBLIC_HOST`
  - Host placed into returned session URLs on the single public manager port
- `MANAGER_PORT`
  - Port for the session manager API
- `CHARACTER_ENG_MANAGER_TOKEN`
  - Optional bearer token for the manager API
- `CHARACTER`
  - Default character when `MODE=app`
- `VISION=1`
  - Enable vision when `MODE=app`
- `START_PAUSED=1`
  - Start runtime paused when `MODE=app`

## Runtime overrides used by the session manager

Each spawned user session gets its own env override set:

- `CHARACTER_ENG_BRIDGE_ENABLED=true`
- `CHARACTER_ENG_BRIDGE_PORT=<unique port>`
- `CHARACTER_ENG_BRIDGE_TOKEN=<unique token>`
- `CHARACTER_ENG_DASHBOARD_ENABLED=true`
- `CHARACTER_ENG_DASHBOARD_PORT=<same bridge port>`
- `CHARACTER_ENG_VISION_ENABLED=true|false`
- `CHARACTER_ENG_VISION_PORT=<unique port>`
- `CHARACTER_ENG_VISION_URL=http://127.0.0.1:<unique port>`

The spawned bridge and vision ports stay internal to the pod/container. Remote users only hit the manager port, and the session manager proxies tokenized browser traffic to the right runtime.

## Local smoke examples

Single shared control plane:

```bash
docker build -t character-eng .
docker run --gpus all -p 7870:7870 -e MODE=session-manager -e PUBLIC_HOST=127.0.0.1 character-eng
```

Single direct runtime:

```bash
docker run --gpus all -p 7862:7862 -e MODE=app -e CHARACTER=greg character-eng
```

Session manager API:

```bash
curl -X POST http://127.0.0.1:7870/sessions \
  -H 'Content-Type: application/json' \
  -d '{"character":"greg","vision":false}'
```

The response includes a tokenized browser URL on the same manager port, for example `http://127.0.0.1:7870/?token=...`.

For faster local heavy-runtime debugging before another RunPod deploy:

```bash
./scripts/vision_smoke.sh
TEARDOWN=1 ./scripts/vision_smoke.sh
```

That command reuses the current heavy-stack startup path, exits once `/model_status` is ready, and prints the raw model-status payload so you can debug `vllm`/vision startup locally before paying for another cold pod boot.

## Lite RunPod validation

Use `Dockerfile.manager-lite` plus `deploy/runpod-lite.toml` when you want to validate the control plane before paying the startup cost of the full GPU image.

If your image is private, configure RunPod registry auth in the TOML and provide the referenced env vars before `up`. `deploy/runpod.py` will reuse an existing RunPod registry auth with the same name or create it on demand.

Suggested build:

```bash
export GHCR_USERNAME=<github-user>
export GHCR_PASSWORD=$(gh auth token)
docker build -f Dockerfile.manager-lite -t ghcr.io/<org>/character-eng:runpod-lite .
docker push ghcr.io/<org>/character-eng:runpod-lite
uv run python deploy/runpod.py --config deploy/runpod-lite.toml up
```

For the full image, keep a non-zero `volume_in_gb` mounted at `/workspace`. The container now uses
`/workspace/character-eng-runtime` for the heavy vision runtime environment and model caches, so a
persistent volume avoids re-installing the full CUDA / vLLM / vision stack on every redeploy.

## GCP VM path

There is now a sibling GCP path under [`deploy/gcp/`](gcp/README.md).

That flow keeps the existing container/runtime architecture and swaps only the infrastructure layer:

- Compute Engine GPU VM instead of a RunPod pod
- persistent disk for runtime state
- optional Caddy TLS termination in front of the manager port

This is the better fit when you want a more deterministic long-lived host and are willing to manage a VM directly.

For a local-direct prompt/frontend loop against hosted heavy services, use:

```bash
./scripts/run_hot_remote.sh
./scripts/stop_hot_remote.sh
```

That path keeps mic/cam local, tunnels the remote vision and Pocket-TTS services over SSH, and leaves the existing local `run_hot.sh` behavior unchanged.
