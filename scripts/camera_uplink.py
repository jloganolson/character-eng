#!/usr/bin/env python3
from __future__ import annotations

import argparse
import signal
import sys
import time
import urllib.error
import urllib.request

import cv2


def _post_json(url: str, payload: bytes) -> None:
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5):
        pass


def _post_jpeg(url: str, payload: bytes) -> None:
    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/octet-stream"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5):
        pass


def _wait_for_service(base_url: str, timeout_s: float) -> None:
    deadline = time.time() + timeout_s
    health_url = f"{base_url.rstrip('/')}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"vision service did not become ready within {timeout_s:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture local webcam frames and push them to a remote vision service.")
    parser.add_argument("--service-url", required=True, help="Base URL for the vision service, e.g. http://127.0.0.1:17860")
    parser.add_argument("--device", default="0", help="OpenCV camera device index or path")
    parser.add_argument("--fps", type=float, default=2.2, help="Upload cadence in frames/sec")
    parser.add_argument("--width", type=int, default=960, help="Capture width")
    parser.add_argument("--height", type=int, default=540, help="Capture height")
    parser.add_argument("--jpeg-quality", type=int, default=72, help="JPEG quality 1-100")
    parser.add_argument("--startup-timeout", type=float, default=30.0, help="Seconds to wait for the remote service")
    args = parser.parse_args()

    base_url = args.service_url.rstrip("/")
    try:
        _wait_for_service(base_url, args.startup_timeout)
    except Exception as exc:
        print(f"camera uplink: {exc}", file=sys.stderr, flush=True)
        return 1

    device = int(args.device) if str(args.device).isdigit() else args.device
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"camera uplink: unable to open camera device {args.device}", file=sys.stderr, flush=True)
        return 1

    if args.width > 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height > 0:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    running = True

    def _handle_signal(signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    quality = max(1, min(100, int(args.jpeg_quality)))
    frame_interval = 1.0 / max(0.2, float(args.fps))

    try:
        _post_json(f"{base_url}/set_input_mode", b'{"mode":"external"}')
    except Exception:
        pass

    print(
        f"camera uplink: streaming local device {args.device} to {base_url} at ~{1.0 / frame_interval:.1f} fps",
        flush=True,
    )

    last_log = 0.0
    sent_frames = 0
    while running:
        loop_started = time.time()
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.1)
            continue
        ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not ok:
            time.sleep(0.1)
            continue
        try:
            _post_jpeg(f"{base_url}/inject_frame", jpeg.tobytes())
            sent_frames += 1
            if time.time() - last_log >= 5.0:
                print(f"camera uplink: sent {sent_frames} frame(s)", flush=True)
                last_log = time.time()
        except urllib.error.URLError as exc:
            print(f"camera uplink: upload failed: {exc}", file=sys.stderr, flush=True)
            time.sleep(0.5)
            continue
        except Exception as exc:
            print(f"camera uplink: unexpected upload error: {exc}", file=sys.stderr, flush=True)
            time.sleep(0.5)
            continue

        sleep_for = frame_interval - (time.time() - loop_started)
        if sleep_for > 0:
            time.sleep(sleep_for)

    try:
        _post_json(f"{base_url}/set_input_mode", b'{"mode":"camera"}')
    except Exception:
        pass
    cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
