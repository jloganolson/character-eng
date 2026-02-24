from unittest.mock import patch

from character_eng.serve import get_local_models, kill_port


def test_get_local_models():
    """get_local_models returns only non-hidden local entries."""
    models = get_local_models()
    for key, cfg in models:
        assert cfg.get("local") is True
        assert not cfg.get("hidden")


@patch("character_eng.serve.subprocess.run")
def test_kill_port_no_lsof(mock_run):
    """kill_port handles missing lsof gracefully."""
    mock_run.side_effect = FileNotFoundError
    # Should not raise
    kill_port("8000")
