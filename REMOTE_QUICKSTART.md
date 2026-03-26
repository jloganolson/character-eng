# Remote Quickstart

This guide is for a collaborator using the shared GCP-backed remote path on a fresh machine.

It covers:

- macOS
- Linux
- Windows via WSL2

This path keeps the app local and uses the shared GCP VM for the heavy stack.
You do not need a local `.env` with provider API keys for this mode.

## What You Need

- repo access
- `gcloud` installed
- Python 3.12+
- `uv` installed
- access already granted with `./deploy/gcp/onboard_user.sh`
- a modern browser

## Important Windows Note

Use WSL2 with Ubuntu and run everything from the WSL shell.
Do not use native PowerShell or `cmd.exe` for the bash scripts in this repo.

Clone the repo into the WSL filesystem, not a mounted Windows drive.
Good: `~/src/character-eng`
Avoid: `/mnt/c/...`

Your browser can still be regular Windows Chrome or Edge and open `http://localhost:7862`.

## One-Time Setup

### 1. Clone the repo

```bash
git clone <repo-url>
cd character-eng
```

### 2. Install local tools

#### macOS

Install:

- Python 3.12+
- `uv`
- Google Cloud CLI (`gcloud`)

Your preferred install path is fine. Homebrew is the usual macOS option.

#### Ubuntu / Debian Linux

Install:

- Python 3.12+
- `uv`
- Google Cloud CLI (`gcloud`)

If you already have Python 3.12, the fastest `uv` install is:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For `gcloud`, use the normal Google Cloud CLI install path for your distro, then restart your shell.

#### Windows

1. Install WSL2.
2. Install Ubuntu from the Microsoft Store.
3. Open the Ubuntu shell.
4. Inside Ubuntu, follow the Linux setup above.

After installing the tools, verify:

```bash
python3 --version
uv --version
gcloud --version
```

### 3. Install repo dependencies

```bash
uv sync
```

You do not need local provider API keys for the shared remote path.

### 4. Log into GCP

```bash
gcloud auth login
gcloud config set project character-eng
```

If your browser login flow signs into the wrong Google account, fix that before continuing.

### 5. Shared Config

The shared remote scripts now fall back automatically to [deploy/gcp.shared-remote.env](deploy/gcp.shared-remote.env).

That file already contains the non-secret shared GCP settings plus the `GCP_SECRET_*` refs for the shared Secret Manager values.
Do not replace those refs with real API keys for the normal collaborator workflow.
The shared remote config also enables nightly VM auto-shutdown at `02:00` in `America/Los_Angeles`.

Only create `deploy/gcp.env` if you need a machine-local override.

## Verify Access

Run:

```bash
./deploy/gcp/doctor.sh
```

Healthy output should show:

- `vm status: RUNNING` or a valid stopped VM state
- `os login: TRUE`
- a manager URL
- a LiveKit URL
- `ssh: ssh-ok` when the VM is running
- `remote hot env: remote-hot-env-ok` when the VM is running

If `doctor.sh` fails with a permission error, your Google account is not allowlisted correctly yet.

## Start The Shared Remote Path

Run:

```bash
./scripts/run_hot_remote_webrtc.sh
```

What happens:

- the script starts the shared GCP VM if needed
- waits for remote vision, Pocket-TTS, manager, and LiveKit
- opens an SSH tunnel for the remote services
- syncs the filtered runtime env from the VM
- starts the local app against the remote heavy stack

Once the app is up, open:

```text
http://localhost:7862
```

Use your local browser mic and camera there.

## Stop

Default stop path:

```bash
./scripts/stop_hot_remote_webrtc.sh
```

That closes the local SSH tunnel and stops the GCP VM.

If someone else still needs the VM up:

```bash
STOP_VM=0 ./scripts/stop_hot_remote_webrtc.sh
```

## Daily Use

Typical loop:

```bash
cd character-eng
./deploy/gcp/doctor.sh
./scripts/run_hot_remote_webrtc.sh
```

When done:

```bash
./scripts/stop_hot_remote_webrtc.sh
```

If nobody stops it manually, the shared VM is also configured to shut itself down nightly at `02:00` Pacific.

## Troubleshooting

### `gcloud` opens the wrong account

Check:

```bash
gcloud config get-value account
```

If needed:

```bash
gcloud auth login
gcloud config set project character-eng
```

### `doctor.sh` says `missing env file`

You have not created `deploy/gcp.env` yet, or you are in the wrong repo checkout.

### `doctor.sh` says `remote hot env: missing or unreadable`

The VM bootstrapped incorrectly or the shared remote deployment is stale.
Ask the repo owner to redeploy the GCP startup path.

### `run_hot_remote_webrtc.sh` fails on SSH

You likely do not have OS Login access yet, or you authenticated with the wrong Google account.

### Browser mic or camera does not work

- make sure you opened `http://localhost:7862` from the same machine running the script
- grant browser mic and camera permissions
- retry in Chrome or Edge if your default browser is restrictive

### Windows browser cannot connect to localhost

Run the scripts in WSL2, but open `http://localhost:7862` in the Windows browser.
WSL2 localhost forwarding should handle it automatically on current Windows builds.

## Reference

- Main quickstart: [QUICKSTART.md](QUICKSTART.md)
- Full repo guide: [README.md](README.md)
- GCP deployment notes: [deploy/gcp/README.md](deploy/gcp/README.md)
