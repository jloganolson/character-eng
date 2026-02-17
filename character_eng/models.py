MODELS = {
    "cerebras-llama": {
        "name": "Llama 3.1 8B (Cerebras)",
        "model": "llama3.1-8b",
        "base_url": "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "stream_usage": False,
    },
    # "cerebras-gpt": {
    #     "name": "GPT-OSS 120B (Cerebras)",
    #     "model": "gpt-oss-120b",
    #     "base_url": "https://api.cerebras.ai/v1",
    #     "api_key_env": "CEREBRAS_API_KEY",
    #     "stream_usage": False,
    # },
}

DEFAULT_MODEL = "cerebras-llama"
