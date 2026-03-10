import os
import time

from dotenv import load_dotenv
from openai import OpenAI
from rich.console import Console

from character_eng.models import MODELS, get_fallback_chain
from character_eng.utils import ts as _ts

load_dotenv()

_console = Console()
_chat_trace_hook = None


def _make_chat_client(model_config: dict) -> OpenAI:
    api_key = model_config.get("api_key") or os.environ[model_config["api_key_env"]]
    return OpenAI(api_key=api_key, base_url=model_config["base_url"])


def set_chat_trace_hook(hook):
    global _chat_trace_hook
    previous = _chat_trace_hook
    _chat_trace_hook = hook
    return previous


class ChatSession:
    def __init__(self, system_prompt: str, model_config: dict):
        self.system_prompt = system_prompt
        self._model_config = model_config
        self._client = _make_chat_client(model_config)
        self._messages: list[dict] = [
            {"role": "system", "content": system_prompt},
        ]
        self._tagged_system_indices: dict[str, int] = {}
        self._last_usage = None

        # Build fallback chain
        self._fallback_configs: list[dict] = []
        for key, cfg in MODELS.items():
            if cfg is model_config:
                chain = get_fallback_chain(key)
                self._fallback_configs = chain[1:]  # skip primary
                break

    def send(self, message: str):
        """Send a message and yield streamed text chunks.

        Uses try/finally so partial responses (e.g. from barge-in closing the
        generator mid-stream) still get recorded in message history.
        Automatically falls back to backup provider on connection/server errors
        (both at stream creation and during iteration before any chunks yield).
        """
        self._messages.append({"role": "user", "content": message})
        started_at = time.time()
        prompt_messages = [dict(msg) for msg in self._messages]

        stream = self._open_stream_with_fallback()

        full_response = []
        try:
            for chunk in stream:
                if chunk.usage:
                    self._last_usage = chunk.usage
                if chunk.choices and chunk.choices[0].delta.content:
                    text = chunk.choices[0].delta.content
                    full_response.append(text)
                    yield text
        finally:
            response_text = "".join(full_response)
            if response_text:
                self._messages.append({"role": "assistant", "content": response_text})
                if _chat_trace_hook is not None:
                    try:
                        _chat_trace_hook({
                            "label": "chat",
                            "provider": self._model_config["name"],
                            "model": self._model_config["model"],
                            "messages": prompt_messages,
                            "started_at": started_at,
                            "finished_at": time.time(),
                            "output": response_text,
                        })
                    except Exception:
                        pass

    def _open_stream_with_fallback(self):
        """Try primary + fallbacks, returning the first working stream.

        Catches errors both at .create() time and on the first iteration
        (streaming APIs often defer the real HTTP call until first read).
        """
        configs = [(self._model_config, self._client)]
        for fb in self._fallback_configs:
            configs.append((fb, _make_chat_client(fb)))

        for i, (cfg, client) in enumerate(configs):
            if i == 0:
                _console.print(f"[dim]  {_ts()} → {cfg['name']} (chat)[/dim]")
            else:
                _console.print(f"[yellow]  {_ts()} → fallback: {cfg['name']} (chat)[/yellow]")

            stream = self._create_stream(cfg, client)
            if stream is None:
                continue

            # Probe the first chunk — streaming errors often surface here
            try:
                first = next(iter(stream))
            except StopIteration:
                return iter([])  # empty stream
            except Exception as e:
                status = getattr(e, "status_code", None)
                if status and 400 <= status < 500:
                    raise
                _console.print(f"[red]  → {cfg['name']} failed on first chunk: {e}[/red]")
                continue

            return self._prepend(first, stream)

        # All failed
        self._messages.pop()
        raise ConnectionError("All LLM providers failed")

    @staticmethod
    def _prepend(first_chunk, stream):
        """Re-attach the probed first chunk to the stream."""
        yield first_chunk
        yield from stream

    def _create_stream(self, config, client):
        """Try to create a streaming completion; return None on provider failure."""
        kwargs = dict(
            model=config["model"],
            messages=self._messages,
            stream=True,
        )
        if config["stream_usage"]:
            kwargs["stream_options"] = {"include_usage": True}
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            status = getattr(e, "status_code", None)
            if status and 400 <= status < 500:
                raise  # auth/client errors shouldn't fallback
            _console.print(f"[red]  → {config['name']} failed: {e}[/red]")
            return None

    def add_assistant(self, content: str):
        """Add an assistant message to history without making an LLM call."""
        self._messages.append({"role": "assistant", "content": content})

    def replace_last_assistant(self, content: str):
        """Replace the most recent assistant message (for barge-in truncation)."""
        for i in range(len(self._messages) - 1, -1, -1):
            if self._messages[i]["role"] == "assistant":
                self._messages[i]["content"] = content
                return

    def inject_system(self, content: str):
        """Append a system message to the conversation history."""
        self._messages.append({"role": "system", "content": content})

    def upsert_system(self, tag: str, content: str):
        """Replace or append a tagged system message without duplicating it."""
        index = self._tagged_system_indices.get(tag)
        if index is not None and 0 <= index < len(self._messages):
            message = self._messages[index]
            if message.get("role") == "system":
                message["content"] = content
                return
        self._messages.append({"role": "system", "content": content})
        self._tagged_system_indices[tag] = len(self._messages) - 1

    def remove_tagged_system(self, tag: str) -> None:
        """Remove a tagged system message if it exists."""
        index = self._tagged_system_indices.pop(tag, None)
        if index is None or not (0 <= index < len(self._messages)):
            return
        message = self._messages[index]
        if message.get("role") != "system":
            return
        self._messages.pop(index)
        for key, value in list(self._tagged_system_indices.items()):
            if value > index:
                self._tagged_system_indices[key] = value - 1

    def get_history(self) -> list[dict]:
        """Return a copy of the message history."""
        return list(self._messages)

    def trace_info(self) -> dict:
        """Return diagnostic info about the session."""
        info = {
            "model": f"{self._model_config['model']} ({self._model_config['name']})",
            "system_prompt": self.system_prompt,
            "history_length": len(self._messages),
        }
        if self._last_usage:
            info["prompt_tokens"] = self._last_usage.prompt_tokens
            info["response_tokens"] = self._last_usage.completion_tokens
            info["total_tokens"] = self._last_usage.total_tokens
        return info
