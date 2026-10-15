from __future__ import annotations

from typing import Any

import pytest

from generator.client import GeneratorConfig, VertexGeminiClient


class _GenerateContentConfig:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class _Types:
    GenerateContentConfig = _GenerateContentConfig


class _Response:
    text = "ok"
    candidates: list[Any] = []
    usage_metadata = None

    @staticmethod
    def to_json_dict() -> dict[str, Any]:
        return {"text": "ok"}


class _Part:
    def __init__(
        self,
        text: str,
        *,
        thought: bool = False,
        thought_signature: str | None = None,
    ) -> None:
        self.text = text
        self.thought = thought
        self.thought_signature = thought_signature


class _Content:
    def __init__(self, parts: list[_Part]) -> None:
        self.parts = parts


class _Candidate:
    finish_reason = "STOP"

    def __init__(self, parts: list[_Part]) -> None:
        self.content = _Content(parts)


@pytest.mark.asyncio
async def test_vertex_gemini_sends_billing_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    labels = {"app": "mpf-parser", "env": "stg"}
    client = VertexGeminiClient(
        GeneratorConfig(model="gemini-test", provider="vertex_gemini", labels=labels)
    )
    captured: dict[str, Any] = {}

    class _Models:
        @staticmethod
        def generate_content(**kwargs: Any) -> _Response:
            captured.update(kwargs)
            return _Response()

    class _Client:
        models = _Models()

        @staticmethod
        def close() -> None:
            return None

    async def _run_inline(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(client, "_get_sdk", lambda: (None, _Types))
    monkeypatch.setattr(client, "_build_client", _Client)
    monkeypatch.setattr("generator.client.asyncio.to_thread", _run_inline)

    await client._call_once([{"role": "user", "content": "hello"}], schema=None)

    request_config = captured["config"]
    assert request_config.kwargs["labels"] == labels


@pytest.mark.asyncio
async def test_vertex_gemini_extracts_readable_thought_summary_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = VertexGeminiClient(
        GeneratorConfig(model="gemini-test", provider="vertex_gemini")
    )

    class _Usage:
        prompt_token_count = 10
        candidates_token_count = 4
        thoughts_token_count = 7
        total_token_count = 21

    class _ThoughtResponse:
        text = '{"tool":"request_action_batch","arguments":{}}'
        candidates = [
            _Candidate(
                [
                    _Part("readable summary", thought=True),
                    _Part(
                        '{"tool":"request_action_batch","arguments":{}}',
                        thought_signature="opaque-signature",
                    ),
                ]
            )
        ]
        usage_metadata = _Usage()

        @staticmethod
        def to_json_dict() -> dict[str, Any]:
            return {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {"text": "readable summary", "thought": True},
                                {
                                    "text": '{"tool":"request_action_batch","arguments":{}}',
                                    "thought_signature": "opaque-signature",
                                },
                            ],
                        }
                    }
                ]
            }

    class _Models:
        @staticmethod
        def generate_content(**kwargs: Any) -> _ThoughtResponse:
            return _ThoughtResponse()

    class _Client:
        models = _Models()

        @staticmethod
        def close() -> None:
            return None

    async def _run_inline(function: Any, *args: Any, **kwargs: Any) -> Any:
        return function(*args, **kwargs)

    monkeypatch.setattr(client, "_get_sdk", lambda: (None, _Types))
    monkeypatch.setattr(client, "_build_client", _Client)
    monkeypatch.setattr("generator.client.asyncio.to_thread", _run_inline)

    response = await client._call_once(
        [{"role": "user", "content": "hello"}], schema=None
    )

    assert response.reasoning_content == "readable summary"
    assert response.reasoning_signature_present is True
    assert response.usage["reasoning_tokens"] == 7
