"""LiveKit AEC — WebRTC AEC3 via livekit.rtc.AudioProcessingModule.

WebRTC's AEC3 algorithm with nonlinear echo suppression.

Requirements: pip install livekit
"""

from livekit.rtc import AudioProcessingModule, AudioFrame


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

    def reset(self) -> None:
        """Reset state between test cases."""
        self._play_buf.clear()
        self._cap_buf.clear()
        # Recreate APM (no reset method available)
        self._apm = AudioProcessingModule(
            echo_cancellation=True,
            noise_suppression=self._noise_suppression,
            high_pass_filter=True,
            auto_gain_control=False,
        )
        self._apm.set_stream_delay_ms(0)

    def close(self) -> None:
        """Clean up."""
        self._play_buf.clear()
        self._cap_buf.clear()
        self._apm = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
