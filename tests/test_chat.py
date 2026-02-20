from unittest.mock import MagicMock, patch

from character_eng.chat import ChatSession

TEST_CONFIG = {
    "name": "Test Model",
    "model": "test-model",
    "base_url": "https://test.example.com/v1",
    "api_key_env": "GEMINI_API_KEY",
    "stream_usage": True,
}


@patch("character_eng.chat.OpenAI")
def test_init_creates_system_message(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    history = session.get_history()
    assert len(history) == 1
    assert history[0] == {"role": "system", "content": "You are Greg."}


@patch("character_eng.chat.OpenAI")
def test_add_assistant_appends_message(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.add_assistant("Hey, what's up?")
    history = session.get_history()
    assert len(history) == 2
    assert history[1] == {"role": "assistant", "content": "Hey, what's up?"}


@patch("character_eng.chat.OpenAI")
def test_inject_system_appends_message(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.inject_system("Something happened.")
    history = session.get_history()
    assert len(history) == 2
    assert history[1] == {"role": "system", "content": "Something happened."}


@patch("character_eng.chat.OpenAI")
def test_send_appends_messages_and_yields_chunks(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)

    # Build mock streaming response: two content chunks + one usage-only chunk
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello "
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "world!"
    chunk2.usage = None

    chunk3 = MagicMock()
    chunk3.choices = []
    chunk3.usage = MagicMock(prompt_tokens=10, completion_tokens=5, total_tokens=15)

    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    chunks = list(session.send("Hi"))

    assert chunks == ["Hello ", "world!"]

    history = session.get_history()
    assert len(history) == 3  # system + user + assistant
    assert history[1] == {"role": "user", "content": "Hi"}
    assert history[2] == {"role": "assistant", "content": "Hello world!"}
    assert session._last_usage.total_tokens == 15


@patch("character_eng.chat.OpenAI")
def test_send_without_stream_usage(mock_openai_cls):
    """When stream_usage is False, stream_options should not be passed."""
    config = {**TEST_CONFIG, "stream_usage": False}
    session = ChatSession("You are Greg.", config)

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hi"
    chunk1.usage = None

    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.return_value = iter([chunk1])

    list(session.send("Hello"))

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    assert "stream_options" not in call_kwargs


@patch("character_eng.chat.OpenAI")
def test_get_history_returns_copy(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    history = session.get_history()
    history.append({"role": "user", "content": "extra"})
    assert len(session.get_history()) == 1  # original unchanged


@patch("character_eng.chat.OpenAI")
def test_trace_info_without_usage(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    info = session.trace_info()
    assert "test-model" in info["model"]
    assert "Test Model" in info["model"]
    assert info["system_prompt"] == "You are Greg."
    assert info["history_length"] == 1
    assert "total_tokens" not in info


@patch("character_eng.chat.OpenAI")
def test_trace_info_with_usage(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150
    session._last_usage = mock_usage

    info = session.trace_info()
    assert info["prompt_tokens"] == 100
    assert info["response_tokens"] == 50
    assert info["total_tokens"] == 150


LOCAL_CONFIG = {
    "name": "LFM-2 2.6B (Local)",
    "model": "lfm-2.6b",
    "base_url": "http://localhost:8000/v1",
    "api_key": "local",
    "stream_usage": False,
    "local": True,
    "path": "/tmp/fake-model",
}


@patch("character_eng.chat.OpenAI")
def test_init_with_direct_api_key(mock_openai_cls):
    """Local model configs use api_key directly instead of api_key_env."""
    session = ChatSession("You are Greg.", LOCAL_CONFIG)
    mock_openai_cls.assert_called_once_with(
        api_key="local",
        base_url="http://localhost:8000/v1",
    )
    assert session.get_history()[0] == {"role": "system", "content": "You are Greg."}


@patch("character_eng.chat.OpenAI")
def test_send_partial_response_recorded_on_generator_close(mock_openai_cls):
    """Closing the send() generator mid-stream should still record partial response."""
    session = ChatSession("You are Greg.", TEST_CONFIG)

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Hello "
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "world!"
    chunk2.usage = None

    chunk3 = MagicMock()
    chunk3.choices = [MagicMock()]
    chunk3.choices[0].delta.content = " More text"
    chunk3.usage = None

    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2, chunk3])

    # Only consume first two chunks, then close the generator (simulates barge-in)
    gen = session.send("Hi")
    first = next(gen)
    second = next(gen)
    gen.close()  # Barge-in — closes generator mid-stream

    assert first == "Hello "
    assert second == "world!"

    # Partial response should still be recorded in history
    history = session.get_history()
    assert len(history) == 3  # system + user + assistant
    assert history[1] == {"role": "user", "content": "Hi"}
    assert history[2] == {"role": "assistant", "content": "Hello world!"}
