# Vision Service

Standalone vision service for character-eng. Provides camera capture, face tracking, person tracking, VLM questioning, and structured JSON endpoints.

## Setup

```bash
cd services/vision
uv sync
```

This creates a separate `.venv` with heavy GPU dependencies (torch, SAM3, InsightFace, torchreid) — fully isolated from the main character-eng venv.

## Run

### Manual start

```bash
# Vision service only (trackers auto-start)
uv run python app.py --auto-start-trackers

# With VLM (requires vLLM running on port 8000)
uv run python app.py --auto-start-trackers --vllm-url http://localhost:8000/v1

# Full stack (vLLM + app)
./start.sh
```

### Auto-start from character-eng

```bash
# From project root — vision service auto-starts if not running
uv run -m character_eng --vision
```

## API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check |
| `/snapshot` | GET | Current visual state as JSON (faces, persons, objects, VLM answers) |
| `/set_questions` | POST | Set constant/ephemeral VLM questions |
| `/set_sam_targets` | POST | Set constant/ephemeral SAM3 tracking targets |
| `/` | GET | Debug UI (camera feed, tracker overlays, VLM slots) |

### Snapshot format

```json
{
  "faces": [{"identity": "Person 1", "bbox": [x, y, w, h], "age": 30, "gender": "M", "confidence": 0.95, "gaze_direction": "at camera", "looking_at_camera": true}],
  "persons": [{"identity": "Person 1", "bbox": [x, y, w, h], "confidence": 0.8}],
  "objects": [{"label": "cup", "bbox": [x, y, w, h], "confidence": 0.7}],
  "vlm_answers": [{"question": "How many people?", "answer": "One", "elapsed": 0.5, "slot_type": "constant"}],
  "timestamp": 1234567890.0
}
```

## GPU budget (~10.5GB)

| Component | VRAM |
|-----------|------|
| SAM3 | ~3.4GB |
| InsightFace buffalo_s | ~0.8GB |
| ReID ResNet50 | ~0.25GB |
| LFM2-VL-3B (vLLM) | ~6GB |

## Important

Run this service from its own directory or via `uv run --project services/vision`. It has its own `pyproject.toml` and `.venv` — the heavy GPU deps (torch, SAM3, insightface) are intentionally isolated from the main character-eng venv to avoid dependency conflicts.
