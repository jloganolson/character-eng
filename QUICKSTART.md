# Quickstart

This is the short operator guide for the two supported working modes:

- `full local`: app + heavy stack on the same machine
- `hybrid GCP`: app local, heavy stack on GCP

Use this instead of reading the full `README.md` when you just want to get running.

If you are a collaborator joining the shared GCP remote path on a fresh machine, use [REMOTE_QUICKSTART.md](/home/logan/Projects/character-eng-worktrees/gcp-envs/REMOTE_QUICKSTART.md).

## Pick a mode

Choose `full local` if:

- you have a strong local GPU box
- you want the lowest latency
- you want mic/cam/speaker to stay fully local

Choose `hybrid GCP` if:

- you want to keep prompt/frontend iteration local
- you want to borrow the hosted heavy stack
- you are okay with the heavy stack living on the GCP VM

## Full Local

Recommended one-command entrypoint:

```bash
./scripts/run_local.sh
```

What it does:

- clean restart of the local app
- starts local voice + vision + dashboard
- keeps media fully local

Useful local commands:

```bash
./scripts/run_heavy.sh      # keep vLLM / vision / Pocket hot
./scripts/run_hot.sh        # restart only the fast app/UI layer
./scripts/vision_smoke.sh   # heavy stack readiness check
./scripts/stop_local.sh     # stop local app + heavy services
```

If you already have the heavy stack up and want the fast loop:

```bash
./scripts/run_heavy.sh
./scripts/run_hot.sh
```

## Hybrid GCP

This is the default hosted-heavy workflow.

Preflight:

```bash
./deploy/gcp/doctor.sh
```

Start:

```bash
./scripts/run_hot_remote_webrtc.sh
```

Stop:

```bash
./scripts/stop_hot_remote_webrtc.sh
```

Direct VM control:

```bash
./deploy/gcp/vm_ctl.sh deploy/gcp.env start
./deploy/gcp/vm_ctl.sh deploy/gcp.env stop
./deploy/gcp/vm_ctl.sh deploy/gcp.env status
```

What this mode does:

- starts the GCP VM if needed
- uses remote vision + remote Pocket-TTS
- keeps the app local
- uses LiveKit for browser mic/cam transport

## Onboarding Another User

Allowlist a Gmail user:

```bash
./deploy/gcp/onboard_user.sh teammate@gmail.com
```

Remove access:

```bash
./deploy/gcp/offboard_user.sh teammate@gmail.com
```

What onboarding grants:

- VM start/stop/status access
- OS Login for SSH
- service account usage needed for the current SSH flow

What the onboarded user should do:

```bash
gcloud auth login
gcloud config set project character-eng
./deploy/gcp/doctor.sh
./scripts/run_hot_remote_webrtc.sh
```

Fresh-machine setup for macOS, Linux, and Windows WSL2 lives in [REMOTE_QUICKSTART.md](/home/logan/Projects/character-eng-worktrees/gcp-envs/REMOTE_QUICKSTART.md).
That guide uses the committed shared remote config at [deploy/gcp.shared-remote.env](/home/logan/Projects/character-eng-worktrees/gcp-envs/deploy/gcp.shared-remote.env), and the remote scripts fall back to it automatically when `deploy/gcp.env` is absent.

## Notes

- `hybrid GCP` is the supported remote path.
- The VM bills while running. If you are done, stop it.
- The shared remote config also schedules a nightly VM shutdown at `02:00` Pacific as a backstop.

## Where To Look Next

- Full repo docs: [README.md](/home/logan/Projects/character-eng-worktrees/gcp-envs/README.md)
- GCP-specific details: [deploy/gcp/README.md](/home/logan/Projects/character-eng-worktrees/gcp-envs/deploy/gcp/README.md)
