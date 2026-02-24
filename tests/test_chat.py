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


@patch("character_eng.chat.OpenAI")
def test_replace_last_assistant(mock_openai_cls):
    """replace_last_assistant should modify the most recent assistant message."""
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.add_assistant("Full response here.")
    session.inject_system("Some eval note.")
    session.replace_last_assistant("Full resp —")
    history = session.get_history()
    assert history[1] == {"role": "assistant", "content": "Full resp —"}
    assert history[2] == {"role": "system", "content": "Some eval note."}
