# GCP Deployment Notes

This is the supported remote deployment path.

The intended first deployment shape is:

- a single Compute Engine GPU VM
- one persistent data disk mounted for `/workspace`-style runtime state
- the existing container running with `MODE=full`
- optional Caddy on the VM for HTTPS termination in front of port `7870`

## Why this path is smaller than the old remote path

Most of the app is already portable:

- the container image
- `deploy/container_entrypoint.sh`
- `character_eng.session_manager`
- `scripts/run_heavy.sh`
- the `services/vision` stack

The older remote deployment path has been removed. The active infrastructure path is now GCP.

## Important browser constraint

If you want browser mic/camera to work remotely, you need HTTPS.

That means the real production path on GCP should be:

- point a public DNS name at the VM
- set `CADDY_DOMAIN` and `CADDY_EMAIL`
- let Caddy terminate TLS and reverse-proxy to `127.0.0.1:7870`

For quick testing, `CADDY_DOMAIN` can also be an `sslip.io` hostname that resolves to the VM IP, for example `136-109-169-132.sslip.io`.

If you omit Caddy and run plain HTTP on the VM IP, CLI checks still work, but browser media capture will not behave like the HTTPS flow because the page is not a secure context.

## Current repo files

- `deploy/gcp.env.example`
  - local config template; copy to `deploy/gcp.env`
- `deploy/gcp.shared-remote.env`
  - committed shared non-secret config for collaborators using the existing shared GCP remote path
- `deploy/gcp/create_vm.sh`
  - local helper that creates the runtime disk, firewall rule, optional static IP, and VM
- `deploy/gcp/startup.sh`
  - instance startup script that mounts the runtime disk, configures Docker, configures NVIDIA container runtime, optionally configures Caddy, and starts the container

## Quick start

1. Copy the env file:

```bash
cp deploy/gcp.env.example deploy/gcp.env
```

2. Fill in at least:

- `GCP_PROJECT_ID`
- `GCP_REGION`
- `GCP_ZONE`
- `CONTAINER_IMAGE`
- either the direct API keys you need, or the matching `GCP_SECRET_*` refs

`deploy/gcp.env` can source your local `.env`. When you create or update the VM, the deploy helper packages the app runtime keys from that environment into the generated VM env file, so the hosted manager/container does not rely on each allowlisted engineer having those keys locally.

The current example defaults to Google's GPU-ready `ml-images/common-cu128-ubuntu-2204-nvidia-570` image family so the VM starts with drivers already in place.

Keep `CHARACTER_ENG_SHARED_VISION_URL=http://127.0.0.1:7860` for the full-container path so per-session runtimes reuse the shared vision stack instead of trying to boot a second private copy.
Set `POCKET_HOST=0.0.0.0` on GCP so Pocket-TTS is reachable over the SSH-tunneled hybrid path.
You can also set `AUTO_SHUTDOWN_ENABLED`, `AUTO_SHUTDOWN_TIME`, and `AUTO_SHUTDOWN_TIMEZONE` to install a VM-local nightly shutdown timer from the startup script.

For shared `remote_hot_webrtc`, prefer storing provider credentials in GCP Secret Manager and setting the corresponding `GCP_SECRET_*` refs in `deploy/gcp.env`.
On VM boot, the startup script resolves those refs into `/etc/character-eng.env` for the container and writes a filtered `/etc/character-eng.remote-hot.env` that `./scripts/run_hot_remote_webrtc.sh` syncs back before launching the local app.
That keeps the repo-side remote workflow off local `.env` files for collaborators who have been allowlisted onto the VM.
If you are onboarding collaborators onto the already-running shared remote environment, commit the non-secret shared settings in `deploy/gcp.shared-remote.env`.
The collaborator-facing remote scripts fall back to `deploy/gcp.shared-remote.env` automatically when `deploy/gcp.env` is absent, so a local `deploy/gcp.env` is only needed for overrides.

For app-only fixes after the heavy base image is already proven, you can layer a fast hotfix image with `Dockerfile.gcp-hotfix` instead of rebuilding the full CUDA environment from scratch.

3. If you want real browser access, also set:

- `PUBLIC_HOST`
- `CADDY_DOMAIN`
- `CADDY_LIVEKIT_DOMAIN`
- `CADDY_EMAIL`

For the WebRTC-backed remote-hot path, `CADDY_LIVEKIT_DOMAIN` is the important one. The local dashboard can stay on `localhost`, but the browser still needs a secure `wss://` LiveKit signaling URL and public RTC ports on the VM.

4. Create the VM:

```bash
./deploy/gcp/create_vm.sh
```

For a dry run of the exact `gcloud` commands:

```bash
DRY_RUN=1 ./deploy/gcp/create_vm.sh
```

## User onboarding

To allowlist another engineer by Gmail address without using the Cloud Console:

```bash
./deploy/gcp/onboard_user.sh teammate@gmail.com
```

That grants:

- a project custom role for VM start/stop/status
- `roles/compute.osLogin`
- `roles/iam.serviceAccountUser` on the VM's attached service account

To remove access later:

```bash
./deploy/gcp/offboard_user.sh teammate@gmail.com
```

For the onboarded user, the preflight check is:

```bash
./deploy/gcp/doctor.sh
```

That prints the current account, instance status, manager/LiveKit URLs, and attempts an SSH check when the VM is running.

## Remote hot loop

Primary hybrid path for the browser-media flow with the app local and the heavy services on GCP:

- `./scripts/run_hot_remote_webrtc.sh`
  - starts the GCP VM if needed
  - waits for remote vision, Pocket-TTS, and LiveKit
  - opens an SSH tunnel for remote vision + Pocket-TTS
  - syncs the filtered runtime env from the VM when available
  - runs the local app in `remote_hot_webrtc` against the remote LiveKit server
- `./scripts/stop_hot_remote_webrtc.sh`
  - closes the SSH tunnel
  - stops the VM by default (`STOP_VM=0` keeps the VM running)

This is the preferred hosted-heavy workflow. It keeps `full local` untouched while using remote heavy services plus remote LiveKit for hybrid use.
The shared remote config in this repo enables a nightly VM shutdown at `02:00` in `America/Los_Angeles` as a billing backstop.

Set a real `LIVEKIT_API_KEY` / `LIVEKIT_API_SECRET` pair in `deploy/gcp.env`, or point `GCP_SECRET_LIVEKIT_API_KEY` / `GCP_SECRET_LIVEKIT_API_SECRET` at Secret Manager values, before creating or updating the VM.
The create script still rejects the old `devkey` / `secret` placeholders, and the LiveKit secret should be at least 32 characters long.

## Registry choice

Recommended option:

- Artifact Registry
  - set `CONTAINER_IMAGE=<region>-docker.pkg.dev/<project>/<repo>/<image>:<tag>`
  - leave registry credentials blank and let the VM service account pull

## GPU base image reality

The provided startup script assumes:

- the VM has an attached GPU
- NVIDIA drivers are already present or can be installed successfully

If you use a plain Ubuntu image, the script will fail fast when `nvidia-smi` is missing. That is intentional. For predictable startup, the longer-term GCP path should graduate to either:

- a GPU-ready base image, or
- a custom Compute Engine image with drivers and container runtime already prepared

## Human steps still required

From this machine, we can script most of the repo-side and `gcloud` parts. You still need to handle:

- attaching billing to the GCP project before Compute Engine and Artifact Registry can be enabled
- creating or updating the DNS record if you want HTTPS browser access
- possibly selecting the final GPU/machine family you want to pay for on GCP
