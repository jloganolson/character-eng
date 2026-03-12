# Local Stack Benchmark Runbook

This branch (`perf-test`) has a repeatable GPU benchmark for comparing the heavy stack across machines.

## Steps

1. **Install deps** (if first time):
   ```bash
   uv sync
   cd services/vision && uv sync && cd ../..
   ```

2. **Start the heavy stack in no-camera mode**:
   ```bash
   VISION_NO_CAMERA=1 ./scripts/run_heavy.sh
   ```
   Wait for it to report all models ready (vLLM, SAM3, face, person). This can take 1-2 minutes on first boot.

3. **Run the benchmark**:
   ```bash
   uv run -m character_eng.local_stack_benchmark
   ```
   This injects prerecorded frames from `services/vision/test_set/`, runs 4 scenarios (vision_poll, vision_query, pocket_tts, stack_overlap) with 1 warmup + 5 measured runs each, and saves a JSON report to `logs/`.

4. **Copy the report back** for comparison. The file will be at:
   ```
   logs/local_stack_benchmark_YYYYMMDD_HHMMSS.json
   ```

## Comparing reports

To diff two reports side by side:
```bash
uv run -m character_eng.local_stack_benchmark compare <report_A.json> <report_B.json>
```

## What it measures

| Scenario | What |
|---|---|
| `vision_poll` | Snapshot + raw frame + annotated frame fetch latency |
| `vision_query` | VLM question-answering end-to-end (ask + poll for result) |
| `pocket_tts` | First-audio latency and full synthesis time |
| `stack_overlap` | All three above running concurrently to expose GPU contention |

## Notes

- The heavy stack must be running before the benchmark starts. The benchmark does NOT start services itself.
- `VISION_NO_CAMERA=1` is critical — it skips webcam init so frames come only from the test set, making results reproducible.
- If the app (port 7862) is also running, you'll get a warning. For clean numbers, only run the heavy stack.
- Pocket-TTS needs a voice file at `voices/greg.safetensors` (check `run_heavy.sh` defaults).
