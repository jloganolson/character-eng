import os

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/Models"))

MODELS = {
    "cerebras-llama": {
        "name": "Llama 3.1 8B (Cerebras)",
        "model": "llama3.1-8b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "stream_usage": False,
        "fallback": "groq-llama-8b",
    },
    "groq-llama-8b": {
        "name": "Llama 3.1 8B (Groq)",
        "model": "llama-3.1-8b-instant",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "stream_usage": False,
        "hidden": True,
    },
    "kimi-k2": {
        "name": "Kimi K2 (Groq)",
        "model": "moonshotai/kimi-k2-instruct",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "stream_usage": False,
        "fallback": "groq-llama-8b",
    },
    "groq-llama": {
        "name": "Llama 3.3 70B (Groq)",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "stream_usage": False,
    },
    "cerebras-gpt": {
        "name": "GPT-OSS 120B (Cerebras)",
        "model": "gpt-oss-120b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "stream_usage": False,
        "hidden": True,
    },
    "groq-gpt": {
        "name": "GPT-OSS 120B (Groq)",
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "stream_usage": False,
        "hidden": True,
    },
    "gemini-3-flash": {
        "name": "Gemini 3.0 Flash (Google)",
        "model": "gemini-3-flash-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "stream_usage": False,
        "hidden": True,
    },
    "gemini-3-flash-lite": {
        "name": "Gemini 3.1 Flash Lite (Google)",
        "model": "gemini-3.1-flash-lite-preview",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "stream_usage": False,
    },
    "gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash (Google)",
        "model": "gemini-2.5-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "stream_usage": False,
        "hidden": True,
    },
    # Local models — need vLLM server running (see serve.py)
    "lfm-2.6b": {
        "name": "LFM-2 2.6B (Local)",
        "model": "lfm-2.6b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "local",
        "stream_usage": False,
        "local": True,
        "hidden": True,
        "path": f"{MODELS_DIR}/LFM-2-2.6B",
    },
    "lfm-1.2b": {
        "name": "LFM-2.5 1.2B (Local)",
        "model": "lfm-1.2b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "local",
        "stream_usage": False,
        "local": True,
        "hidden": True,
        "path": f"{MODELS_DIR}/LFM-2.5-1.2B",
    },
}

# ── Active model selection ──────────────────────────────────────
# Two-tier routing: CHAT_MODEL for streaming dialogue, MICRO_MODEL for
# parallel microservices (script_check, thought, director, beat, reconcile,
# expression, condition). Both use Groq for speed.
CHAT_MODEL = os.environ.get("CHARACTER_ENG_CHAT_MODEL", "kimi-k2")          # K2 — streaming chat (15/16 roles, 196ms TTFT)
MICRO_MODEL = os.environ.get("CHARACTER_ENG_MICRO_MODEL", "groq-llama-8b")   # 8B — microservices (14/16 roles, ~300ms structured JSON)
BIG_MODEL = os.environ.get("CHARACTER_ENG_BIG_MODEL", "groq-llama")        # kept for API compat (no longer required)

# Legacy alias used by QA scripts
DEFAULT_MODEL = CHAT_MODEL


def get_fallback_chain(model_key: str) -> list[dict]:
    """Return [primary_config, fallback_config, ...] for a model key."""
    chain = []
    seen = set()
    key = model_key
    while key and key not in seen:
        seen.add(key)
        cfg = MODELS[key]
        chain.append(cfg)
        key = cfg.get("fallback")
    return chain
