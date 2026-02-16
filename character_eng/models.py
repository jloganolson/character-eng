MODELS = {
    "gemini": {
        "name": "Gemini Flash",
        "model": "gemini-2.0-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "api_key_env": "GEMINI_API_KEY",
        "stream_usage": True,
    },
    "cerebras": {
        "name": "Llama 70B (Cerebras)",
        "model": "llama-3.3-70b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "stream_usage": False,
    },
}

DEFAULT_MODEL = "gemini"
