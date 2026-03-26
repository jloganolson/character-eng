# character-eng archive-debug workflow

When the current repo is `character-eng`, prefer this flow:

1. Run:
   `./scripts/archive_debug.sh <archive-ref-or-path> "<issue summary>"`
2. If the issue is naming-related, also run:
   `python3 scripts/debug_archive_naming.py <archive-path>`
3. If the issue is a structured vision/state mutation, convert the earliest bad event into:
   `tests/vision_snippets/fixtures/moments/...`
4. Validate with:
   `uv run pytest tests/test_qa_vision_snippets.py`

Interpretation guide:

- `user_transcript_final` right, `response_done` right, `people_state` wrong:
  state write-path bug
- `vision_state_update` already wrong:
  vision interpreter bug or missing sanitizer
- `reconcile` introduces the wrong fact:
  reconcile bug or missing sanitizer
- archive tool shows many `speculative_*` flags:
  state quality drift from VLM phrasing is likely worth a guard or fixture
