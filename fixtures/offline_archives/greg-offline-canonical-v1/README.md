This directory stores the canonical offline dashboard archive fixture.

- source capture: `history/sessions/962fef15`
- installed local id: `greg-offline-canonical-v1`
- intended use: `./scripts/run_hot_offline.sh`

Do not point the dashboard at this directory directly. Install it into local `history/sessions/` with:

```bash
python3 scripts/install_offline_archive_fixture.py
```

The installer rewrites machine-local manifest paths and normalizes the archive id/title for the current checkout.
