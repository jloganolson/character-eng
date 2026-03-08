from __future__ import annotations

import re
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from playwright.sync_api import Error, Page, expect, sync_playwright

from character_eng.qa_full_stack import TurnRecord, _write_report


@pytest.fixture()
def report_url(tmp_path: Path) -> str:
    json_path = tmp_path / "report.json"
    html_path = tmp_path / "report.html"
    turn = TurnRecord(
        scene_name="approach",
        image_prompt="test prompt",
        visual_summary="A hooded passerby slows down at Greg's stand.",
        scripted_user_line="What's the pitch?",
        stt_transcript="What's the pitch?",
        assistant_response="Free water. Free advice. Interested?",
        assistant_audio_ms=820,
        response_ttft_ms=210,
        response_total_ms=980,
        trace_events=[],
    )
    _write_report(
        [turn],
        json_path,
        html_path,
        session_events=[
            {
                "type": "vision_poll",
                "timestamp": 1.0,
                "seq": 1,
                "data": {
                    "faces": 0,
                    "persons": 1,
                    "objects": 1,
                    "object_labels": ["bottle"],
                    "vlm_answers": ["A person is near the table."],
                },
            },
            {
                "type": "stt_result",
                "timestamp": 1.2,
                "seq": 1,
                "data": {"transcript": "What's the pitch?"},
            },
            {
                "type": "user_turn_start",
                "timestamp": 1.25,
                "seq": 2,
                "data": {"text": "What's the pitch?"},
            },
            {
                "type": "response_done",
                "timestamp": 1.8,
                "seq": 2,
                "data": {"ttft_ms": 210, "total_ms": 980},
            },
            {
                "type": "assistant_tts",
                "timestamp": 2.1,
                "seq": 3,
                "data": {"audio_ms": 820},
            },
        ],
    )

    handler = partial(SimpleHTTPRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/{html_path.name}"
    finally:
        server.shutdown()
        thread.join(timeout=2)
        server.server_close()


@pytest.fixture()
def browser_page() -> Page:
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1600, "height": 1100})
            yield page
            browser.close()
    except Error as exc:
        pytest.skip(f"Playwright browser not available: {exc}")


def test_report_interactions_in_browser(browser_page: Page, report_url: str):
    page = browser_page
    page.goto(report_url)

    expect(page.locator("h1")).to_have_text("Full Stack QA Trace Report")
    expect(page.locator("#detail-type")).to_have_text("vision_poll")

    response_card = page.locator(".stream-card", has_text="Assistant done")
    response_card.get_by_role("button", name="Note").click()
    expect(page.locator("#detail-type")).to_have_text("response_done")
    expect(page.locator("#detail-label")).to_contain_text("Assistant done")
    assert page.locator("#detail-related button").count() >= 1

    page.locator("#detail-note").fill("Keep this kind of timing tight.")
    expect(response_card.get_by_role("button", name="Note")).to_have_class(
        re.compile(r"\bhas-note\b")
    )

    page.select_option("#density-mode", "compact")
    expect(page.locator(".page-shell")).to_have_class(
        re.compile(r"\bcompact-mode\b")
    )

    page.locator("#show-chronology").uncheck()
    expect(page.locator("#detail-chronology")).to_have_class(
        re.compile(r"\bhidden\b")
    )
    page.locator("#show-chronology").check()
    expect(page.locator(".chronology-row.selected")).to_have_count(1)
