from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch


sys.modules.setdefault("cv2", types.SimpleNamespace(COLOR_BGR2RGB=0, cvtColor=lambda frame, _: frame))


class _FakeImageModule:
    @staticmethod
    def fromarray(frame):
        class _Image:
            def __init__(self, array):
                self.size = (array.shape[1], array.shape[0])

        return _Image(frame)


sys.modules.setdefault("PIL", types.SimpleNamespace(Image=_FakeImageModule))

VISION_DIR = Path(__file__).resolve().parent.parent / "services" / "vision"
if str(VISION_DIR) not in sys.path:
    sys.path.insert(0, str(VISION_DIR))

from person_tracker import PersonTracker


class _FakeNativeSam3Processor:
    def set_image(self, image, state=None):
        state = dict(state or {})
        state["image_size"] = image.size
        return state

    def set_text_prompt(self, prompt: str, state):
        prompt_state = dict(state)
        prompt_state["masks"] = torch.tensor(
            [[[[0, 1], [1, 1]]]],
            dtype=torch.bool,
        )
        prompt_state["boxes"] = torch.tensor([[1, 2, 6, 9]], dtype=torch.float32)
        prompt_state["scores"] = torch.tensor([0.91], dtype=torch.float32)
        prompt_state["prompt"] = prompt
        return prompt_state


def test_person_tracker_supports_native_sam3_processor_api():
    tracker = PersonTracker(
        cam=None,
        sam3_getter=lambda: (object(), _FakeNativeSam3Processor()),
        face_getter=lambda: [],
        device="cpu",
        dtype=None,
    )
    frame = np.zeros((12, 12, 3), dtype=np.uint8)

    results = tracker._run_sam3_all(frame, ["person", "cup"])

    assert sorted(results) == ["cup", "person"]
    assert results["person"][0]["bbox"] == (1, 2, 5, 7)
    assert results["person"][0]["confidence"] == pytest.approx(0.91)
    assert results["person"][0]["label"] == "person"
    assert results["person"][0]["mask"].dtype == np.bool_
