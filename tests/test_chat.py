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
def test_respond_uses_existing_history_without_appending_fake_user(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.add_assistant("Earlier reply")

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock()]
    chunk1.choices[0].delta.content = "Fresh "
    chunk1.usage = None

    chunk2 = MagicMock()
    chunk2.choices = [MagicMock()]
    chunk2.choices[0].delta.content = "response"
    chunk2.usage = None

    mock_client = mock_openai_cls.return_value
    mock_client.chat.completions.create.return_value = iter([chunk1, chunk2])

    chunks = list(session.respond())

    assert chunks == ["Fresh ", "response"]
    history = session.get_history()
    assert history == [
        {"role": "system", "content": "You are Greg."},
        {"role": "assistant", "content": "Earlier reply"},
        {"role": "assistant", "content": "Fresh response"},
    ]


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


@patch("character_eng.chat.OpenAI")
def test_replace_last_assistant_removes_message_when_content_empty(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.add_assistant("Tiny fragment")
    session.inject_system("Runtime note")

    session.replace_last_assistant("")

    history = session.get_history()
    assert history == [
        {"role": "system", "content": "You are Greg."},
        {"role": "system", "content": "Runtime note"},
    ]


@patch("character_eng.chat.OpenAI")
def test_upsert_system_replaces_tagged_message_without_duplication(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.upsert_system("runtime_turn_guardrails", "First guardrail")
    session.add_assistant("hello")
    session.upsert_system("runtime_turn_guardrails", "Updated guardrail")

    history = session.get_history()
    system_messages = [message for message in history if message["role"] == "system"]

    assert len(system_messages) == 2
    assert system_messages[1] == {"role": "system", "content": "Updated guardrail"}


@patch("character_eng.chat.OpenAI")
def test_remove_tagged_system_clears_runtime_context(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.upsert_system("runtime_turn_guardrails", "Guardrail")
    session.add_assistant("hello")
    session.remove_tagged_system("runtime_turn_guardrails")

    history = session.get_history()
    assert history == [
        {"role": "system", "content": "You are Greg."},
        {"role": "assistant", "content": "hello"},
    ]


@patch("character_eng.chat.OpenAI")
def test_rollback_last_turn_removes_latest_user_and_assistant(mock_openai_cls):
    session = ChatSession("You are Greg.", TEST_CONFIG)
    session.add_assistant("First reply")
    session.inject_system("Runtime note")
    session.add_assistant("Second reply")
    session.inject_system("Another note")
    session._messages.append({"role": "user", "content": "half sentence"})
    session._messages.append({"role": "assistant", "content": "partial reply"})

    session.rollback_last_turn()

    history = session.get_history()
    assert history[-1] == {"role": "system", "content": "Another note"}
    assert {"role": "user", "content": "half sentence"} not in history
    assert {"role": "assistant", "content": "partial reply"} not in history
