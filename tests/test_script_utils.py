from __future__ import annotations

import pytest

from scripts._utils import build_generator


def _model_config(**overrides: object) -> dict:
    config = {
        "model": "test-model",
        "conversation_protocol": "openai_chat",
        "api_key": "test-key",
    }
    config.update(overrides)
    return config


def test_build_generator_resolves_base_url_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("ADAEXPLOREBENCH_TEST_BASE_URL", "  https://example.test/v1  ")

    client = build_generator(
        _model_config(base_url_env="ADAEXPLOREBENCH_TEST_BASE_URL")
    )

    assert client.config.base_url == "https://example.test/v1"


def test_build_generator_requires_configured_base_url_environment(
    monkeypatch,
) -> None:
    monkeypatch.delenv("ADAEXPLOREBENCH_MISSING_BASE_URL", raising=False)

    with pytest.raises(ValueError, match="ADAEXPLOREBENCH_MISSING_BASE_URL"):
        build_generator(
            _model_config(base_url_env="ADAEXPLOREBENCH_MISSING_BASE_URL")
        )


def test_build_generator_keeps_responses_reasoning_summary_setting() -> None:
    client = build_generator(
        _model_config(
            conversation_protocol="openai_responses",
            api_mode="responses_background",
            reasoning_summary="auto",
        )
    )

    assert client.config.reasoning_summary == "auto"
