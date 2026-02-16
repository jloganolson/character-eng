from unittest.mock import MagicMock, patch

from character_eng.chat import ChatSession


@patch("character_eng.chat.OpenAI")
def test_init_creates_system_message(mock_openai_cls):
    session = ChatSession("You are Greg.")
    history = session.get_history()
    assert len(history) == 1
    assert history[0] == {"role": "system", "content": "You are Greg."}


@patch("character_eng.chat.OpenAI")
def test_inject_system_appends_message(mock_openai_cls):
    session = ChatSession("You are Greg.")
    session.inject_system("Something happened.")
    history = session.get_history()
    assert len(history) == 2
    assert history[1] == {"role": "system", "content": "Something happened."}


@patch("character_eng.chat.OpenAI")
def test_get_history_returns_copy(mock_openai_cls):
    session = ChatSession("You are Greg.")
    history = session.get_history()
    history.append({"role": "user", "content": "extra"})
    assert len(session.get_history()) == 1  # original unchanged


@patch("character_eng.chat.OpenAI")
def test_trace_info_without_usage(mock_openai_cls):
    session = ChatSession("You are Greg.")
    info = session.trace_info()
    assert info["model"] == "gemini-2.0-flash"
    assert info["system_prompt"] == "You are Greg."
    assert info["history_length"] == 1
    assert "total_tokens" not in info


@patch("character_eng.chat.OpenAI")
def test_trace_info_with_usage(mock_openai_cls):
    session = ChatSession("You are Greg.")
    mock_usage = MagicMock()
    mock_usage.prompt_tokens = 100
    mock_usage.completion_tokens = 50
    mock_usage.total_tokens = 150
    session._last_usage = mock_usage

    info = session.trace_info()
    assert info["prompt_tokens"] == 100
    assert info["response_tokens"] == 50
    assert info["total_tokens"] == 150
