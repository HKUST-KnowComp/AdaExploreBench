from __future__ import annotations

import asyncio

import httpx

from generator.client import GeneratorConfig, GeneratorError, OpenAICompatClient


class _Response:
    def __init__(self, value: dict, status_code: int = 200) -> None:
        self._value = value
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict:
        return self._value

    def raise_for_status(self) -> None:
        return None


class _Store:
    def __init__(self, pending: str | None = None) -> None:
        self.pending = pending
        self.recorded: list[tuple[str, str]] = []
        self.cleared: list[tuple[str, str]] = []

    def pending_response_id(self, request_key: str) -> str | None:
        return self.pending

    async def record_pending_response(self, request_key: str, response_id: str) -> None:
        self.pending = response_id
        self.recorded.append((request_key, response_id))

    async def clear_pending_response(self, request_key: str, response_id: str) -> None:
        self.pending = None
        self.cleared.append((request_key, response_id))

def _completed(response_id: str = "resp_1") -> dict:
    return {
        "id": response_id,
        "status": "completed",
        "model": "gpt-test",
        "usage": {"input_tokens": 3, "output_tokens": 4, "total_tokens": 7},
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"ok":true}'}],
            }
        ],
    }


def _client() -> OpenAICompatClient:
    return OpenAICompatClient(
        GeneratorConfig(
            model="gpt-test",
            base_url="https://api.openai.com/v1",
            api_key="test-key",
            api_mode="responses_background",
            output_format="json_schema",
            max_completion_tokens=48000,
            effort="high",
            reasoning_summary="auto",
            background_poll_interval_seconds=0.1,
        )
    )


def test_responses_payload_uses_background_structured_output_fields() -> None:
    payload = _client()._build_responses_payload(
        [{"role": "user", "content": "generate"}],
        {"type": "object", "properties": {}, "additionalProperties": False},
    )

    assert payload["background"] is True
    assert payload["max_output_tokens"] == 48000
    assert payload["reasoning"] == {"effort": "high", "summary": "auto"}
    assert payload["text"]["format"]["type"] == "json_schema"
    assert "messages" not in payload
    assert "max_completion_tokens" not in payload
    assert "response_format" not in payload


def test_background_submission_is_persisted_then_polled(monkeypatch) -> None:
    requests: list[tuple[str, str]] = []
    replies = [
        _Response({"id": "resp_1", "status": "queued"}),
        _Response(_completed()),
    ]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def request(self, method, url, **kwargs):
            requests.append((method, url))
            return replies.pop(0)

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("generator.client.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("generator.client.asyncio.sleep", no_sleep)
    client = _client()
    store = _Store()
    client.set_background_store(store)

    response = asyncio.run(
        client.generate(messages=[{"role": "user", "content": "generate"}], schema={})
    )

    assert response.content == '{"ok":true}'
    assert [method for method, _ in requests] == ["POST", "GET"]
    assert store.recorded[0][1] == "resp_1"
    assert store.cleared[0][1] == "resp_1"


def test_background_resume_and_poll_transport_retry_never_resubmits(monkeypatch) -> None:
    requests: list[tuple[str, str]] = []
    replies = [httpx.RemoteProtocolError("disconnect"), _Response(_completed("resp_saved"))]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def request(self, method, url, **kwargs):
            requests.append((method, url))
            reply = replies.pop(0)
            if isinstance(reply, Exception):
                raise reply
            return reply

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("generator.client.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("generator.client.asyncio.sleep", no_sleep)
    client = _client()
    store = _Store("resp_saved")
    client.set_background_store(store)

    response = asyncio.run(
        client.generate(messages=[{"role": "user", "content": "generate"}], schema={})
    )

    assert response.content == '{"ok":true}'
    assert requests == [
        ("GET", "https://api.openai.com/v1/responses/resp_saved"),
        ("GET", "https://api.openai.com/v1/responses/resp_saved"),
    ]
    assert store.recorded == []
    assert store.cleared[0][1] == "resp_saved"


def test_background_transport_retry_stops_at_configured_limit(monkeypatch) -> None:
    requests = 0

    class FakeClient:
        async def request(self, method, url, **kwargs):
            nonlocal requests
            requests += 1
            raise httpx.ConnectError("offline")

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("generator.client.asyncio.sleep", no_sleep)
    client = _client()
    client.config.request_retry_limit = 2

    try:
        asyncio.run(
            client._background_request_json(
                FakeClient(),
                "POST",
                "https://api.openai.com/v1/responses",
                {},
                payload={},
                trace_state={"attempt_count": 0, "retry_events": []},
            )
        )
    except GeneratorError as exc:
        assert "2 attempts" in str(exc)
        assert "ConnectError" in str(exc)
    else:
        raise AssertionError("background transport retry did not stop")
    assert requests == 2


def test_background_attempt_timeout_retries_complete_call(monkeypatch) -> None:
    attempts = 0
    client = _client()
    client.config.request_timeout_seconds = 0.01
    client.config.request_retry_limit = 2

    async def never_finishes(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(1)

    monkeypatch.setattr(client, "_call_responses_background", never_finishes)
    trace_state = {"attempt_count": 0, "retry_events": []}

    try:
        asyncio.run(
            client._call_with_retry(
                [{"role": "user", "content": "generate"}],
                {},
                trace_state=trace_state,
            )
        )
    except GeneratorError as exc:
        assert "exceeded 1 seconds" in str(exc)
    else:
        raise AssertionError("background attempt timeout did not stop")
    assert attempts == 2
    assert [item["event"] for item in trace_state["retry_events"]] == [
        "background_attempt_timeout",
        "background_attempt_timeout",
    ]


def test_chat_completions_retries_cloudflare_524(monkeypatch) -> None:
    requests = 0
    replies = [
        _Response({}, status_code=524),
        _Response({
            "model": "gpt-test",
            "usage": {},
            "choices": [{
                "message": {"content": '{"ok":true}'},
                "finish_reason": "stop",
            }],
        }),
    ]

    class FakeClient:
        def __init__(self, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args) -> None:
            return None

        async def post(self, url, **kwargs):
            nonlocal requests
            requests += 1
            return replies.pop(0)

    async def no_sleep(delay: float) -> None:
        return None

    monkeypatch.setattr("generator.client.httpx.AsyncClient", FakeClient)
    monkeypatch.setattr("generator.client.asyncio.sleep", no_sleep)
    client = _client()
    client.config.api_mode = "chat_completions"
    client.config.request_retry_limit = 2

    response = asyncio.run(
        client.generate(messages=[{"role": "user", "content": "JSON"}], schema={})
    )

    assert requests == 2
    assert response.content == '{"ok":true}'


def test_parse_response_reads_vllm_reasoning_key():
    vllm_payload = {
        "model": "Qwen/Qwen3.8-27B",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"ok":true}',
                    "reasoning": "thinking out loud",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    response = OpenAICompatClient._parse_response(vllm_payload)
    assert response.reasoning_content == "thinking out loud"

    openai_payload = {
        "model": "gpt-x",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": '{"ok":true}',
                    "reasoning_content": "chain of thought",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {},
    }
    response = OpenAICompatClient._parse_response(openai_payload)
    assert response.reasoning_content == "chain of thought"


def test_parse_responses_response_reads_reasoning_summary():
    payload = _completed()
    payload["output"].insert(
        0,
        {
            "id": "rs_1",
            "type": "reasoning",
            "summary": [
                {"type": "summary_text", "text": "First summary section."},
                {"type": "summary_text", "text": "Second summary section."},
            ],
        },
    )

    response = OpenAICompatClient._parse_responses_response(payload)

    assert response.content == '{"ok":true}'
    assert response.reasoning_content == (
        "First summary section.\n\nSecond summary section."
    )


def test_reasoning_summary_requires_responses_api_mode():
    try:
        GeneratorConfig(
            model="gpt-test",
            api_mode="chat_completions",
            reasoning_summary="auto",
        )
    except ValueError as exc:
        assert "responses_background" in str(exc)
    else:
        raise AssertionError("chat completions accepted reasoning summaries")
