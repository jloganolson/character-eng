
import torch
from queue import Queue
from transformers.generation.streamers import BaseStreamer

class AudioIteratorStreamer(BaseStreamer):
    """
    Streamer that decodes audio tokens (codes) into PCM waveforms on the fly.
    
    Args:
        speech_tokenizer: The tokenizer instance used to decode codes to audio.
        buffer_size (int): Number of tokens to buffer before decoding (to avoid artifacts).
    """
    def __init__(self, speech_tokenizer, buffer_size=5):
        self.speech_tokenizer = speech_tokenizer
        self.buffer_size = buffer_size
        
        self.token_buffer = []
        self.audio_queue = Queue()
        self.stop_signal = None

    def put(self, value):
        """Standard streamer put (receives 1st codebook tokens)."""
        pass

    def put_audio_codes(self, codec_ids):
        """
        Custom method to receive all codebooks for a time step.
        codec_ids: [batch, num_groups] or [batch, 1, num_groups]
        """
        # Assume batch size 1
        if codec_ids.dim() == 3:
            codes = codec_ids[0, 0]
        else:
            codes = codec_ids[0]
            
        self.token_buffer.append(codes)

        if len(self.token_buffer) >= self.buffer_size:
            self._decode_and_enqueue(self.token_buffer)
            self.token_buffer = []

    def end(self):
        """Signals the end of generation."""
        if self.token_buffer:
            self._decode_and_enqueue(self.token_buffer)
            self.token_buffer = []
        self.audio_queue.put(self.stop_signal)

    def _decode_and_enqueue(self, tokens):
        if not tokens:
            return
            
        # tokens is a list of tensors, each [num_groups]
        device = tokens[0].device
        codes_tensor = torch.stack(tokens, dim=0) # [seq_len, num_groups]
        
        try:
            # speech_tokenizer.decode expects List[Dict["audio_codes": tensor]]
            wavs, sr = self.speech_tokenizer.decode([{"audio_codes": codes_tensor}])
            
            if wavs and len(wavs) > 0:
                self.audio_queue.put((wavs[0], sr))
        except Exception as e:
            print(f"Streaming decode error: {e}")

    def __iter__(self):
        return self

    def __next__(self):
        value = self.audio_queue.get()
        if value == self.stop_signal:
            raise StopIteration()
        return value
