import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL = "gemini-2.0-flash"


class ChatSession:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self._config = types.GenerateContentConfig(
            system_instruction=system_prompt,
        )
        self._chat = self._client.chats.create(
            model=MODEL, config=self._config
        )
        self._last_response = None

    def send(self, message: str):
        """Send a message and yield streamed text chunks."""
        stream = self._chat.send_message_stream(message)
        for chunk in stream:
            if chunk.text:
                yield chunk.text
            self._last_response = chunk

    def trace_info(self) -> dict:
        """Return diagnostic info about the session."""
        info = {
            "model": MODEL,
            "system_prompt": self.system_prompt,
            "history_length": len(self._chat._curated_history),
        }
        if self._last_response and self._last_response.usage_metadata:
            meta = self._last_response.usage_metadata
            info["prompt_tokens"] = meta.prompt_token_count
            info["response_tokens"] = meta.candidates_token_count
            info["total_tokens"] = meta.total_token_count
        return info
