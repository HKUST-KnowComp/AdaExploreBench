from __future__ import annotations

from typing import Any

from generator.client import GeneratorConfig, GoogleGeminiClient
from scripts._utils import build_generator


def test_google_gemini_builds_developer_api_client(monkeypatch) -> None:
    captured: dict[str, Any] = {}

    class _GenAI:
        @staticmethod
        def Client(**kwargs: Any) -> object:
            captured.update(kwargs)
            return object()

    monkeypatch.setenv("GOOGLE_API_KEY", "test-google-key")
    client = GoogleGeminiClient(
        GeneratorConfig(
            model="gemini-test",
            provider="google_gemini",
            api_key_env="GOOGLE_API_KEY",
        )
    )
    monkeypatch.setattr(client, "_get_sdk", lambda: (_GenAI, None))

    client._build_client()

    assert captured["api_key"] == "test-google-key"
    assert captured["http_options"]["api_version"] == "v1beta"
    assert "vertexai" not in captured


def test_build_generator_selects_google_gemini_client() -> None:
    client = build_generator(
        {
            "provider": "google_gemini",
            "conversation_protocol": "gemini_generate_content",
            "model": "gemini-test",
            "api_key": "test-google-key",
        }
    )

    assert isinstance(client, GoogleGeminiClient)
