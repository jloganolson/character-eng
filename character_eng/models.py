import os

MODELS_DIR = os.environ.get("MODELS_DIR", os.path.expanduser("~/Models"))

MODELS = {
    "cerebras-llama": {
        "name": "Llama 3.1 8B (Cerebras)",
        "model": "llama3.1-8b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "stream_usage": False,
    },
    "groq-llama": {
        "name": "Llama 3.3 70B (Groq)",
        "model": "llama-3.3-70b-versatile",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "stream_usage": False,
    },
    "groq-gpt": {
        "name": "GPT-OSS 120B (Groq)",
        "model": "openai/gpt-oss-120b",
        "base_url": "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "stream_usage": False,
    },
    # Local models — need vLLM server running (see serve.py)
    "lfm-2.6b": {
        "name": "LFM-2 2.6B (Local)",
        "model": "lfm-2.6b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "local",
        "stream_usage": False,
        "local": True,
        "path": f"{MODELS_DIR}/LFM-2-2.6B",
    },
    "lfm-1.2b": {
        "name": "LFM-2.5 1.2B (Local)",
        "model": "lfm-1.2b",
        "base_url": "http://localhost:8000/v1",
        "api_key": "local",
        "stream_usage": False,
        "local": True,
        "path": f"{MODELS_DIR}/LFM-2.5-1.2B",
    },
}

DEFAULT_MODEL = "cerebras-llama"
