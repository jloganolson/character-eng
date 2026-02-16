import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = "gemini-2.0-flash"
BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class ChatSession:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self._client = OpenAI(
            api_key=os.environ["GEMINI_API_KEY"],
            base_url=BASE_URL,
        )
        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]
        self._last_usage = None

    def send(self, message: str):
        """Send a message and yield streamed text chunks."""
        self._messages.append({"role": "user", "content": message})

        stream = self._client.chat.completions.create(
            model=MODEL,
            messages=self._messages,
            stream=True,
            stream_options={"include_usage": True},
        )

        full_response = []
        for chunk in stream:
            if chunk.usage:
                self._last_usage = chunk.usage
            if chunk.choices and chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                full_response.append(text)
                yield text

        self._messages.append({"role": "assistant", "content": "".join(full_response)})

    def inject_system(self, content: str):
        """Append a system message to the conversation history."""
        self._messages.append({"role": "system", "content": content})

    def get_history(self) -> list[dict]:
        """Return a copy of the message history."""
        return list(self._messages)

    def trace_info(self) -> dict:
        """Return diagnostic info about the session."""
        info = {
            "model": MODEL,
            "system_prompt": self.system_prompt,
            "history_length": len(self._messages),
        }
        if self._last_usage:
            info["prompt_tokens"] = self._last_usage.prompt_tokens
            info["response_tokens"] = self._last_usage.completion_tokens
            info["total_tokens"] = self._last_usage.total_tokens
        return info
