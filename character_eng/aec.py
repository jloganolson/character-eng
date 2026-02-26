"""Acoustic echo cancellation via LiveKit WebRTC AEC3.

Wraps livekit.rtc.AudioProcessingModule for real-time echo cancellation.
All audio must be 16kHz/int16/mono. Frames are 10ms (160 samples).

Also provides resample_to_16k() for converting speaker output (24kHz/48kHz)
to the 16kHz required by the AEC.
"""

from livekit.rtc import AudioFrame, AudioProcessingModule


class LiveKitAEC:
    """Acoustic echo cancellation via WebRTC AEC3 (LiveKit APM).

    All audio must be 16kHz/int16/mono. Frames must be exactly 10ms (160 samples).
    """

    FRAME_SIZE = 160  # 10ms at 16kHz (mandatory for WebRTC APM)
    FRAME_BYTES = FRAME_SIZE * 2
    SAMPLE_RATE = 16000

    def __init__(self, stream_delay_ms: int = 0, noise_suppression: bool = True):
        """
        Args:
            stream_delay_ms: Estimated delay between speaker output and mic capture.
                             Accounts for device buffer + acoustic propagation.
        """
        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=noise_suppression,
            high_pass_filter=True,
            auto_gain_control=False,
        )
        self._apm.set_stream_delay_ms(stream_delay_ms)
        self._noise_suppression = noise_suppression
        self._play_buf = bytearray()
        self._cap_buf = bytearray()

    def feed_playback(self, pcm_16k: bytes) -> None:
        """Feed speaker reference audio (16kHz int16 mono).

        Buffers into 10ms frames, processes via process_reverse_stream().
        """
        self._play_buf.extend(pcm_16k)
        while len(self._play_buf) >= self.FRAME_BYTES:
            frame_data = bytes(self._play_buf[:self.FRAME_BYTES])
            del self._play_buf[:self.FRAME_BYTES]

            frame = AudioFrame(
                data=frame_data,
                sample_rate=self.SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=self.FRAME_SIZE,
            )
            self._apm.process_reverse_stream(frame)

    def process_capture(self, pcm_16k: bytes) -> bytes:
        """Process mic audio through AEC (16kHz int16 mono).

        Buffers into 10ms frames, processes via process_stream().
        Returns cleaned audio.
        """
        self._cap_buf.extend(pcm_16k)
        out_chunks = []

        while len(self._cap_buf) >= self.FRAME_BYTES:
            frame_data = bytes(self._cap_buf[:self.FRAME_BYTES])
            del self._cap_buf[:self.FRAME_BYTES]

            frame = AudioFrame(
                data=frame_data,
                sample_rate=self.SAMPLE_RATE,
                num_channels=1,
                samples_per_channel=self.FRAME_SIZE,
            )
            self._apm.process_stream(frame)
            # process_stream modifies frame in-place
            out_chunks.append(bytes(frame.data))

        return b"".join(out_chunks)

    def close(self) -> None:
        """Clean up."""
        self._play_buf.clear()
        self._cap_buf.clear()
        self._apm = None


def resample_to_16k(pcm: bytes, source_rate: int) -> bytes:
    """Resample int16 mono PCM from source_rate to 16kHz.

    Uses linear interpolation for non-integer ratios (e.g. 44100→16000),
    simple decimation for integer ratios (e.g. 48000→16000).
    """
    if source_rate == 16000:
        return pcm

    import numpy as np

    samples = np.frombuffer(pcm, dtype=np.int16)
    if len(samples) == 0:
        return pcm

    ratio = source_rate / 16000
    int_ratio = source_rate // 16000
    if source_rate == int_ratio * 16000:
        # Integer ratio (e.g. 48000/16000 = 3x, 24000/16000 = 1.5x fails)
        # Actually 24000/16000 = 1.5 is NOT integer, only 48000/16000 = 3
        return samples[::int_ratio].copy().tobytes()

    # Non-integer ratio: linear interpolation
    old_len = len(samples)
    new_len = round(old_len / ratio)
    if new_len == 0:
        return b""
    old_idx = np.arange(old_len)
    new_idx = np.linspace(0, old_len - 1, new_len)
    return np.interp(new_idx, old_idx, samples.astype(np.float64)).astype(np.int16).tobytes()
