from pathlib import Path

import pytest

from character_eng.rust_eyes_bridge import (
    RustEyesBridgeError,
    RustEyesFaceUpdate,
    RustEyesRuntimeClient,
)


def _write_map(path: Path, body: str) -> None:
    path.write_text(body.strip() + "\n", encoding="utf-8")


def test_resolve_expression_key_errors_on_missing_mapping(tmp_path):
    mapping_path = tmp_path / "expressions.toml"
    _write_map(
        mapping_path,
        """
        [expressions]
        neutral = "neutral"
        """,
    )
    client = RustEyesRuntimeClient(
        base_url="http://127.0.0.1:5555",
        expression_map_path=mapping_path,
    )
    client._available_expression_keys = {"neutral"}

    with pytest.raises(RustEyesBridgeError) as exc:
        client.resolve_expression_key("happy")

    assert "no rust-eyes expression mapping for 'happy'" in str(exc.value)
    assert str(mapping_path) in str(exc.value)


def test_resolve_expression_key_errors_on_unavailable_runtime_key(tmp_path):
    mapping_path = tmp_path / "expressions.toml"
    _write_map(
        mapping_path,
        """
        [expressions]
        happy = "happy_v2"
        """,
    )
    client = RustEyesRuntimeClient(
        base_url="http://127.0.0.1:5555",
        expression_map_path=mapping_path,
    )
    client._available_expression_keys = {"neutral", "excited"}

    with pytest.raises(RustEyesBridgeError) as exc:
        client.resolve_expression_key("happy")

    assert "mapped rust-eyes expression key 'happy_v2'" in str(exc.value)
    assert str(mapping_path) in str(exc.value)


def test_push_face_uses_expression_key_and_runtime_update_payload(tmp_path, monkeypatch):
    mapping_path = tmp_path / "expressions.toml"
    _write_map(
        mapping_path,
        """
        [expressions]
        curious = "curious"
        """,
    )
    client = RustEyesRuntimeClient(
        base_url="http://127.0.0.1:5555",
        owner="character-eng-test",
        ttl=1.25,
        expression_map_path=mapping_path,
    )
    client._available_expression_keys = {"curious"}

    captured = {}

    def fake_request_json(path, *, payload=None):
        captured["path"] = path
        captured["payload"] = payload
        return {"ok": True}

    monkeypatch.setattr(client, "_request_json", fake_request_json)

    response = client.push_face(
        RustEyesFaceUpdate(
            gaze="left table",
            gaze_type="glance",
            expression="curious",
            source="test",
        )
    )

    assert response == {"ok": True}
    assert captured["path"] == "/api/runtime/update"
    assert captured["payload"]["owner"] == "character-eng-test"
    assert captured["payload"]["ttl"] == 1.25
    assert captured["payload"]["expressionKey"] == "curious"
    assert captured["payload"]["gaze"]["lookX"] < 0
    assert captured["payload"]["gaze"]["lookY"] < 0
