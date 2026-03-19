from __future__ import annotations

import asyncio
import io
import threading
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from PIL import Image
from livekit import rtc
from livekit.rtc._proto import video_frame_pb2 as proto_video_frame

from character_eng.transport_metrics import write_metrics


@dataclass(frozen=True)
class LiveKitSession:
    server_url: str
    room_name: str
    token: str
    participant_identity: str
    participant_name: str
    audio_metrics_path: str = ""
    video_metrics_path: str = ""


class LiveKitMediaBridge:
    def __init__(self, session: LiveKitSession, *, video_max_fps: float = 6.0):
        self._session = session
        self._video_max_fps = max(0.5, float(video_max_fps))
        self._on_audio: Callable[[bytes], None] | None = None
        self._on_video_frame: Callable[[bytes], None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._stop_requested = threading.Event()
        self._connected = threading.Event()
        self._room: rtc.Room | None = None
        self._audio_source: rtc.AudioSource | None = None
        self._audio_track: rtc.LocalAudioTrack | None = None
        self._outgoing_audio: asyncio.Queue[bytes] | None = None
        self._stop_async: asyncio.Event | None = None
        self._tasks: set[asyncio.Task] = set()
        self._lock = threading.Lock()
        self._stats = {
            "connected": False,
            "remote_audio_tracks": 0,
            "remote_video_tracks": 0,
            "audio_frames_in": 0,
            "audio_bytes_in": 0,
            "audio_frames_out": 0,
            "audio_bytes_out": 0,
            "video_frames_in": 0,
            "video_frames_sent": 0,
            "last_video_width": 0,
            "last_video_height": 0,
            "last_video_bytes": 0,
            "last_error": "",
        }

    def start(
        self,
        *,
        on_audio: Callable[[bytes], None] | None = None,
        on_video_frame: Callable[[bytes], None] | None = None,
    ) -> None:
        self._on_audio = on_audio
        self._on_video_frame = on_video_frame
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_requested.clear()
        self._connected.clear()
        self._thread = threading.Thread(target=self._thread_main, daemon=True, name="livekit-media-bridge")
        self._thread.start()

    def stop(self) -> None:
        self._stop_requested.set()
        if self._loop is not None and self._stop_async is not None:
            self._loop.call_soon_threadsafe(self._stop_async.set)
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        self._connected.clear()

    def wait_until_connected(self, timeout: float = 8.0) -> bool:
        return self._connected.wait(timeout=timeout)

    def connected(self) -> bool:
        return self._connected.is_set()

    def status_snapshot(self) -> dict:
        with self._lock:
            return dict(self._stats)

    def push_tts_pcm(self, pcm: bytes) -> None:
        if not pcm or self._loop is None or self._outgoing_audio is None:
            return
        with self._lock:
            self._stats["audio_frames_out"] += 1
            self._stats["audio_bytes_out"] += len(pcm)
        write_metrics(self._session.audio_metrics_path or None, {
            "mode": "remote_hot_webrtc",
            "source": "livekit_transport",
            "direction": "egress",
            "audio_frames_in": int(self._stats["audio_frames_in"]),
            "audio_bytes_in": int(self._stats["audio_bytes_in"]),
            "audio_frames_out": int(self._stats["audio_frames_out"]),
            "audio_bytes_out": int(self._stats["audio_bytes_out"]),
            "updated_at": time.time(),
        })
        self._loop.call_soon_threadsafe(self._outgoing_audio.put_nowait, bytes(pcm))

    def clear_tts_audio(self) -> None:
        if self._loop is None or self._audio_source is None:
            return
        self._loop.call_soon_threadsafe(self._audio_source.clear_queue)

    def _thread_main(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._run())
        finally:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            loop.close()
            self._loop = None

    async def _run(self) -> None:
        self._stop_async = asyncio.Event()
        self._outgoing_audio = asyncio.Queue()
        room = rtc.Room(loop=asyncio.get_running_loop())
        self._room = room
        self._audio_source = rtc.AudioSource(sample_rate=24000, num_channels=1, loop=asyncio.get_running_loop())
        self._audio_track = rtc.LocalAudioTrack.create_audio_track("assistant-tts", self._audio_source)

        @room.on("connected")
        def _on_connected() -> None:
            self._connected.set()
            with self._lock:
                self._stats["connected"] = True

        @room.on("disconnected")
        def _on_disconnected(*_args) -> None:
            self._connected.clear()
            with self._lock:
                self._stats["connected"] = False

        @room.on("track_subscribed")
        def _on_track_subscribed(track, _publication, participant) -> None:
            if participant.identity == self._session.participant_identity:
                return
            if isinstance(track, rtc.RemoteAudioTrack):
                with self._lock:
                    self._stats["remote_audio_tracks"] += 1
                task = asyncio.create_task(self._consume_audio(track))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)
            elif isinstance(track, rtc.RemoteVideoTrack):
                with self._lock:
                    self._stats["remote_video_tracks"] += 1
                task = asyncio.create_task(self._consume_video(track))
                self._tasks.add(task)
                task.add_done_callback(self._tasks.discard)

        try:
            await room.connect(self._session.server_url, self._session.token)
            await room.local_participant.publish_track(self._audio_track)
            publish_task = asyncio.create_task(self._pump_outgoing_audio())
            self._tasks.add(publish_task)
            publish_task.add_done_callback(self._tasks.discard)
            await self._stop_async.wait()
        except Exception as exc:
            with self._lock:
                self._stats["last_error"] = str(exc)
        finally:
            self._connected.clear()
            with self._lock:
                self._stats["connected"] = False
            for task in list(self._tasks):
                task.cancel()
            if self._tasks:
                await asyncio.gather(*list(self._tasks), return_exceptions=True)
            if self._room is not None and self._room.isconnected():
                await self._room.disconnect()
            if self._audio_source is not None:
                await self._audio_source.aclose()
            self._room = None
            self._audio_source = None
            self._audio_track = None
            self._outgoing_audio = None
            self._stop_async = None

    async def _pump_outgoing_audio(self) -> None:
        assert self._outgoing_audio is not None
        assert self._audio_source is not None
        while not self._stop_requested.is_set():
            try:
                pcm = await asyncio.wait_for(self._outgoing_audio.get(), timeout=0.5)
            except TimeoutError:
                continue
            if not pcm:
                continue
            frame = rtc.AudioFrame(
                data=pcm,
                sample_rate=24000,
                num_channels=1,
                samples_per_channel=len(pcm) // 2,
            )
            await self._audio_source.capture_frame(frame)

    async def _consume_audio(self, track: rtc.RemoteAudioTrack) -> None:
        stream = rtc.AudioStream.from_track(
            track=track,
            loop=asyncio.get_running_loop(),
            sample_rate=16000,
            num_channels=1,
            frame_size_ms=80,
        )
        try:
            async for event in stream:
                pcm = event.frame.data.cast("B").tobytes()
                with self._lock:
                    self._stats["audio_frames_in"] += 1
                    self._stats["audio_bytes_in"] += len(pcm)
                write_metrics(self._session.audio_metrics_path or None, {
                    "mode": "remote_hot_webrtc",
                    "source": "livekit_transport",
                    "direction": "ingress",
                    "audio_frames_in": int(self._stats["audio_frames_in"]),
                    "audio_bytes_in": int(self._stats["audio_bytes_in"]),
                    "audio_frames_out": int(self._stats["audio_frames_out"]),
                    "audio_bytes_out": int(self._stats["audio_bytes_out"]),
                    "updated_at": time.time(),
                })
                if self._on_audio is not None:
                    try:
                        await asyncio.to_thread(self._on_audio, pcm)
                    except Exception as exc:
                        write_metrics(self._session.audio_metrics_path or None, {
                            "mode": "remote_hot_webrtc",
                            "source": "livekit_transport",
                            "direction": "ingress",
                            "audio_frames_in": int(self._stats["audio_frames_in"]),
                            "audio_bytes_in": int(self._stats["audio_bytes_in"]),
                            "audio_frames_out": int(self._stats["audio_frames_out"]),
                            "audio_bytes_out": int(self._stats["audio_bytes_out"]),
                            "last_error": str(exc),
                            "updated_at": time.time(),
                        })
        finally:
            await stream.aclose()

    async def _consume_video(self, track: rtc.RemoteVideoTrack) -> None:
        stream = rtc.VideoStream.from_track(track=track, loop=asyncio.get_running_loop())
        next_due = 0.0
        try:
            async for event in stream:
                now = time.monotonic()
                with self._lock:
                    self._stats["video_frames_in"] += 1
                if now < next_due:
                    continue
                next_due = now + (1.0 / self._video_max_fps)
                jpeg_bytes, width, height = await asyncio.to_thread(self._frame_to_jpeg, event.frame)
                with self._lock:
                    self._stats["video_frames_sent"] += 1
                    self._stats["last_video_width"] = width
                    self._stats["last_video_height"] = height
                    self._stats["last_video_bytes"] = len(jpeg_bytes)
                write_metrics(self._session.video_metrics_path or None, {
                    "mode": "remote_hot_webrtc",
                    "source": "livekit_transport",
                    "video_frames_in": int(self._stats["video_frames_in"]),
                    "video_frames_sent": int(self._stats["video_frames_sent"]),
                    "last_upload_ms": 0.0,
                    "avg_upload_ms": 0.0,
                    "target_fps": round(self._video_max_fps, 1),
                    "last_frame_bytes": int(self._stats["last_video_bytes"]),
                    "width": int(width),
                    "height": int(height),
                    "updated_at": time.time(),
                })
                if self._on_video_frame is not None:
                    await asyncio.to_thread(self._on_video_frame, jpeg_bytes)
        finally:
            await stream.aclose()

    @staticmethod
    def _frame_to_jpeg(frame: rtc.VideoFrame) -> tuple[bytes, int, int]:
        rgb_frame = frame.convert(proto_video_frame.VideoBufferType.RGB24)
        array = np.frombuffer(rgb_frame.data, dtype=np.uint8).reshape((rgb_frame.height, rgb_frame.width, 3))
        image = Image.fromarray(array, mode="RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=72, optimize=True)
        return buffer.getvalue(), rgb_frame.width, rgb_frame.height
