from __future__ import annotations

import asyncio
import hashlib
import json
import os
from copy import deepcopy
from dataclasses import dataclass, field
from time import monotonic
from typing import Any, Literal

import httpx

from .conversation_adapters import ProviderConversationRequest

Message = dict[str, Any]  # {"role": "user"|"system"|"assistant", "content": "..."}

RETRYABLE_HTTP_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 504, 524})


@dataclass
class GeneratorConfig:
    model: str
    base_url: str = ""
    provider: Literal[
        "openai_compat", "vertex_gemini", "google_gemini"
    ] = "openai_compat"
    conversation_protocol: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    vertex_project: str | None = None
    vertex_project_env: str | None = None
    vertex_location: str | None = None
    vertex_location_env: str | None = None
    max_tokens: int = 2048
    temperature: float | None = None
    top_p: float | None = None
    batch_size: int = 8
    output_format: Literal["text", "json_object", "json_schema"] = "text"
    extra_body: dict = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)
    # Reasoning budget — set in configs/models/*.yaml under the `budget:` key,
    # which is flattened into GeneratorConfig on load.
    thinking_tokens: int | None = None
    max_completion_tokens: int | None = None
    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None = None
    request_timeout_seconds: float = 300.0
    request_retry_limit: int = 6
    request_retry_base_delay_seconds: float = 1.0
    # Runner-level failure policy, shared by the benchmark trajectory and the
    # out-of-band relevant-fields diagnostics: provider errors get this many
    # retries with backoff, and a malformed response gets this many repair
    # rounds with the error reason fed back.
    provider_retry_limit: int = 5
    syntax_repair_limit: int = 1
    api_mode: Literal["chat_completions", "responses_background"] = "chat_completions"
    reasoning_summary: Literal["auto", "concise", "detailed"] | None = None
    background_poll_interval_seconds: float = 2.0

    def __post_init__(self) -> None:
        if (
            self.reasoning_summary is not None
            and self.api_mode != "responses_background"
        ):
            raise ValueError(
                "reasoning_summary requires api_mode='responses_background'"
            )

    def resolve_api_key(self) -> str | None:
        raw: str | None = None
        if self.api_key is not None:
            raw = self.api_key
        elif self.api_key_env is not None:
            raw = os.environ.get(self.api_key_env)

        if raw is None:
            return None

        # Copy/paste from browsers, chats, or docs can introduce smart quotes
        # or invisible Unicode markers that break HTTP header ASCII encoding.
        cleaned = raw.strip()
        cleaned = cleaned.strip("\"'`“”‘’")
        for ch in ("\ufeff", "\u200b", "\u200c", "\u200d", "\u200e", "\u200f", "\u2060", "\u00a0"):
            cleaned = cleaned.replace(ch, "")

        return cleaned

    def resolve_vertex_project(self) -> str | None:
        if self.vertex_project is not None and self.vertex_project.strip():
            return self.vertex_project.strip()
        if self.vertex_project_env is not None:
            raw = os.environ.get(self.vertex_project_env)
            if raw:
                return raw.strip()
        return None

    def resolve_vertex_location(self) -> str | None:
        if self.vertex_location is not None and self.vertex_location.strip():
            return self.vertex_location.strip()
        if self.vertex_location_env is not None:
            raw = os.environ.get(self.vertex_location_env)
            if raw:
                return raw.strip()
        return None


@dataclass
class GeneratorResponse:
    content: str
    model: str
    usage: dict[str, int]
    raw: dict
    reasoning_content: str = ""
    reasoning_signature_present: bool = False
    finish_reason: str = ""
    assistant_content: Any | None = None


class GeneratorError(Exception):
    pass


class OpenAICompatClient:
    """Async OpenAI-compatible LLM client with batch support and exponential backoff."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.batch_size)
        self._trace_recorder = None
        self._background_store = None

    def set_trace_recorder(self, recorder) -> None:
        self._trace_recorder = recorder

    def set_background_store(self, store) -> None:
        """Attach durable response-ID storage for resumable background requests."""
        self._background_store = store

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        messages: list[Message] | ProviderConversationRequest | None = None,
        schema: dict | None = None,
        trace_context: dict[str, Any] | None = None,
        *,
        request: ProviderConversationRequest | None = None,
    ) -> GeneratorResponse:
        if isinstance(messages, ProviderConversationRequest):
            if request is not None:
                raise TypeError("conversation request was supplied twice")
            request = messages
            messages = None
        if request is not None:
            wire_messages = request.messages
        elif messages is not None:
            wire_messages = messages
        else:
            raise TypeError("generate requires messages or a conversation request")
        async with self._semaphore:
            started = monotonic()
            trace_state: dict[str, Any] = {"attempt_count": 0, "retry_events": []}
            try:
                response = await self._call_with_retry(
                    wire_messages, schema, request=request, trace_state=trace_state
                )
            except Exception as exc:
                if self._trace_recorder is not None:
                    await self._trace_recorder.record_failure(
                        provider=self.config.provider,
                        model=self.config.model,
                        messages=wire_messages,
                        schema=schema,
                        error=exc,
                        duration_ms=int((monotonic() - started) * 1000),
                        trace_context=trace_context,
                        attempt_count=trace_state["attempt_count"],
                        retry_events=trace_state["retry_events"],
                    )
                raise
            if self._trace_recorder is not None:
                await self._trace_recorder.record_success(
                    provider=self.config.provider,
                    model=self.config.model,
                    messages=wire_messages,
                    schema=schema,
                    response=response,
                    duration_ms=int((monotonic() - started) * 1000),
                    trace_context=trace_context,
                    attempt_count=trace_state["attempt_count"],
                    retry_events=trace_state["retry_events"],
                )
            return response

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _should_trust_env(self) -> bool:
        base_url = (self.config.base_url or "").strip().lower()
        return "api.openai.com" in base_url

    def _build_payload(
        self,
        messages: list[Message],
        schema: dict | None,
        request: ProviderConversationRequest | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": request.messages if request is not None else messages,
        }

        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature

        if self.config.max_completion_tokens is not None:
            payload["max_completion_tokens"] = self.config.max_completion_tokens
        else:
            payload["max_tokens"] = self.config.max_tokens

        if self.config.top_p is not None:
            payload["top_p"] = self.config.top_p

        # Response format
        if schema is not None and self.config.output_format == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema, "strict": True},
            }
        elif self.config.output_format == "json_object":
            payload["response_format"] = {"type": "json_object"}
        elif self.config.output_format == "json_schema":
            payload["response_format"] = {"type": "json_object"}

        # Reasoning effort (OpenAI o-series)
        if self.config.effort is not None:
            payload["reasoning_effort"] = self.config.effort

        # Thinking token budget / vendor-specific reasoning controls.
        if self.config.thinking_tokens is not None:
            payload["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.config.thinking_tokens,
            }

        # Vendor-specific pass-through fields should be merged into the top-level
        # request body. The OpenAI SDK's `extra_body=` helper does this client-side;
        # providers like DeepSeek do not expect a literal JSON key named
        # `extra_body` in the HTTP payload.
        for k, v in self.config.extra_body.items():
            payload[k] = v

        return payload

    @staticmethod
    def _validate_header_ascii(value: str, field_name: str) -> None:
        try:
            value.encode("ascii")
        except UnicodeEncodeError as exc:
            raise GeneratorError(
                f"{field_name} contains non-ASCII characters after sanitization. "
                "This usually means the copied API key still includes smart quotes "
                "or invisible Unicode characters."
            ) from exc

    async def _call_with_retry(
        self,
        messages: list[Message],
        schema: dict | None,
        request: ProviderConversationRequest | None = None,
        trace_state: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        if self.config.api_mode == "responses_background":
            max_retries = max(1, int(self.config.request_retry_limit))
            timeout_seconds = max(1.0, float(self.config.request_timeout_seconds))
            last_error: GeneratorError | None = None
            for attempt in range(max_retries):
                if trace_state is not None:
                    trace_state["attempt_count"] = attempt + 1
                try:
                    async with asyncio.timeout(timeout_seconds):
                        return await self._call_responses_background(
                            messages, schema, request=request, trace_state=trace_state
                        )
                except TimeoutError:
                    last_error = GeneratorError(
                        "Background response attempt exceeded "
                        f"{timeout_seconds:g} seconds"
                    )
                    if trace_state is not None:
                        trace_state["retry_events"].append(
                            {
                                "attempt": attempt + 1,
                                "event": "background_attempt_timeout",
                            }
                        )
                except GeneratorError as exc:
                    last_error = exc
            raise last_error
        payload = self._build_payload(messages, schema, request=request)
        api_key = self.config.resolve_api_key()
        if not api_key:
            source = self.config.api_key_env or "configured api_key"
            raise GeneratorError(
                f"Missing API key for model '{self.config.model}'. "
                f"Expected a value from {source}."
            )
        headers: dict[str, str] = {"Content-Type": "application/json"}
        auth_value = f"Bearer {api_key}"
        self._validate_header_ascii(auth_value, "Authorization header")
        headers["Authorization"] = auth_value

        url = self.config.base_url.rstrip("/") + "/chat/completions"

        max_retries = max(1, int(self.config.request_retry_limit))
        timeout_seconds = max(1.0, float(self.config.request_timeout_seconds))
        trust_env = self._should_trust_env()
        async with httpx.AsyncClient(timeout=timeout_seconds, trust_env=trust_env) as client:
            for attempt in range(max_retries):
                if trace_state is not None:
                    trace_state["attempt_count"] = attempt + 1
                try:
                    resp = await client.post(url, json=payload, headers=headers)
                    if resp.status_code in (401, 403):
                        body_preview = resp.text[:500].strip()
                        raise GeneratorError(
                            f"Authentication failed with HTTP {resp.status_code}. "
                            f"Check {self.config.api_key_env or 'api_key'}, base_url, and model name. "
                            f"Response body: {body_preview}"
                        )
                    if resp.status_code == 400:
                        body_preview = resp.text[:1000].strip()
                        raise GeneratorError(
                            "Request rejected with HTTP 400 Bad Request. "
                            "This usually means the model/provider does not accept "
                            "one of the request fields (for example response_format, "
                            "max_tokens, or model name). "
                            f"Response body: {body_preview}"
                        )
                    if resp.status_code in RETRYABLE_HTTP_STATUS_CODES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    resp.raise_for_status()
                    try:
                        data = resp.json()
                    except json.JSONDecodeError:
                        body_preview = resp.text[:1000].strip()
                        if attempt == max_retries - 1:
                            raise GeneratorError(
                                "Provider returned a non-JSON response after all retries. "
                                f"Response body: {body_preview}"
                            )
                        await asyncio.sleep(2 ** attempt)
                        continue
                    return self._parse_response(data)
                except httpx.TimeoutException:
                    if trace_state is not None:
                        trace_state["retry_events"].append(
                            {"attempt": attempt + 1, "event": "timeout"}
                        )
                    if attempt == max_retries - 1:
                        raise GeneratorError("Request timed out after all retries")
                    await asyncio.sleep(self.config.request_retry_base_delay_seconds * (2 ** attempt))
                except httpx.TransportError as exc:
                    if trace_state is not None:
                        trace_state["retry_events"].append(
                            {
                                "attempt": attempt + 1,
                                "event": "transport_error",
                                "error_type": exc.__class__.__name__,
                                "message": str(exc),
                            }
                        )
                    if attempt == max_retries - 1:
                        raise GeneratorError(
                            f"Transport error after all retries: {exc.__class__.__name__}: {exc}"
                        ) from exc
                    await asyncio.sleep(self.config.request_retry_base_delay_seconds * (2 ** attempt))

        raise GeneratorError(f"All {max_retries} retries exhausted")

    def _build_responses_payload(
        self,
        messages: list[Message],
        schema: dict | None,
        request: ProviderConversationRequest | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "input": request.messages if request is not None else messages,
            "background": True,
        }
        if self.config.max_completion_tokens is not None:
            payload["max_output_tokens"] = self.config.max_completion_tokens
        else:
            payload["max_output_tokens"] = self.config.max_tokens
        if schema is not None and self.config.output_format == "json_schema":
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": "response",
                    "schema": schema,
                    "strict": True,
                }
            }
        elif self.config.output_format == "json_object":
            payload["text"] = {"format": {"type": "json_object"}}
        reasoning: dict[str, Any] = {}
        if self.config.effort is not None:
            reasoning["effort"] = self.config.effort
        if self.config.reasoning_summary is not None:
            reasoning["summary"] = self.config.reasoning_summary
        if reasoning:
            payload["reasoning"] = reasoning
        for key, value in self.config.extra_body.items():
            payload[key] = value
        return payload

    def _request_headers(self) -> dict[str, str]:
        api_key = self.config.resolve_api_key()
        if not api_key:
            source = self.config.api_key_env or "configured api_key"
            raise GeneratorError(
                f"Missing API key for model '{self.config.model}'. Expected a value from {source}."
            )
        auth_value = f"Bearer {api_key}"
        self._validate_header_ascii(auth_value, "Authorization header")
        return {"Content-Type": "application/json", "Authorization": auth_value}

    async def _background_request_json(
        self,
        client: httpx.AsyncClient,
        method: str,
        url: str,
        headers: dict[str, str],
        *,
        payload: dict[str, Any] | None,
        trace_state: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Retry one background HTTP operation within the configured limit."""
        attempt = 0
        max_attempts = max(1, int(self.config.request_retry_limit))
        while True:
            attempt += 1
            if trace_state is not None:
                trace_state["attempt_count"] = attempt
            try:
                response = await client.request(method, url, json=payload, headers=headers)
                if response.status_code in (401, 403):
                    raise GeneratorError(
                        f"Authentication failed with HTTP {response.status_code}: "
                        f"{response.text[:500].strip()}"
                    )
                if response.status_code == 400:
                    raise GeneratorError(
                        f"Background Responses request rejected with HTTP 400: "
                        f"{response.text[:1000].strip()}"
                    )
                if response.status_code not in RETRYABLE_HTTP_STATUS_CODES:
                    response.raise_for_status()
                    try:
                        value = response.json()
                    except json.JSONDecodeError as exc:
                        raise GeneratorError(
                            "Background Responses endpoint returned non-JSON content: "
                            f"{response.text[:1000].strip()}"
                        ) from exc
                    if not isinstance(value, dict):
                        raise GeneratorError("Background Responses endpoint returned a non-object")
                    return value
                event = f"http_{response.status_code}"
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                event = exc.__class__.__name__
            if trace_state is not None:
                trace_state["retry_events"].append({"attempt": attempt, "event": event})
            if attempt >= max_attempts:
                raise GeneratorError(
                    "Background Responses HTTP operation failed after "
                    f"{attempt} attempts; last event: {event}"
                )
            delay = min(
                30.0,
                self.config.request_retry_base_delay_seconds * (2 ** min(attempt - 1, 5)),
            )
            await asyncio.sleep(delay)

    async def _call_responses_background(
        self,
        messages: list[Message],
        schema: dict | None,
        request: ProviderConversationRequest | None = None,
        trace_state: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        payload = self._build_responses_payload(messages, schema, request=request)
        request_key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
        response_id = None
        if self._background_store is not None:
            response_id = self._background_store.pending_response_id(request_key)

        headers = self._request_headers()
        base_url = self.config.base_url.rstrip("/") + "/responses"
        timeout_seconds = max(1.0, float(self.config.request_timeout_seconds))
        async with httpx.AsyncClient(
            timeout=timeout_seconds, trust_env=self._should_trust_env()
        ) as client:
            if response_id:
                data = await self._background_request_json(
                    client, "GET", f"{base_url}/{response_id}", headers,
                    payload=None, trace_state=trace_state,
                )
            else:
                data = await self._background_request_json(
                    client, "POST", base_url, headers,
                    payload=payload, trace_state=trace_state,
                )
                response_id = str(data.get("id") or "")
                if not response_id:
                    raise GeneratorError("Background Responses submission returned no response ID")
                if self._background_store is not None:
                    await self._background_store.record_pending_response(
                        request_key, response_id
                    )

            while str(data.get("status") or "") in {"queued", "in_progress"}:
                await asyncio.sleep(max(0.1, self.config.background_poll_interval_seconds))
                data = await self._background_request_json(
                    client, "GET", f"{base_url}/{response_id}", headers,
                    payload=None, trace_state=trace_state,
                )

        if self._background_store is not None:
            await self._background_store.clear_pending_response(request_key, response_id)
        status = str(data.get("status") or "")
        if status != "completed":
            detail = data.get("error") or data.get("incomplete_details") or {}
            raise GeneratorError(
                f"Background response {response_id} ended with status {status}: {detail}"
            )
        return self._parse_responses_response(data)

    @staticmethod
    def _parse_responses_response(data: dict[str, Any]) -> GeneratorResponse:
        parts: list[str] = []
        refusals: list[str] = []
        reasoning_summary_parts: list[str] = []
        for item in data.get("output") or []:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "reasoning":
                for summary in item.get("summary") or []:
                    if not isinstance(summary, dict):
                        continue
                    if summary.get("type") == "summary_text":
                        text = str(summary.get("text") or "").strip()
                        if text:
                            reasoning_summary_parts.append(text)
                continue
            if item.get("type") != "message":
                continue
            for content in item.get("content") or []:
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    parts.append(str(content.get("text") or ""))
                elif content.get("type") == "refusal":
                    refusals.append(str(content.get("refusal") or ""))
        if refusals:
            raise GeneratorError("Model refused the structured request: " + " ".join(refusals))
        if not parts:
            raise GeneratorError("Completed background response contained no output text")
        return GeneratorResponse(
            content="".join(parts),
            model=str(data.get("model") or ""),
            usage=data.get("usage") or {},
            raw=data,
            reasoning_content="\n\n".join(reasoning_summary_parts),
            finish_reason="stop",
        )

    @staticmethod
    def _parse_response(data: dict) -> GeneratorResponse:
        choice = data["choices"][0]
        message = choice.get("message", {}) or {}
        raw_content = message.get("content")
        if isinstance(raw_content, list):
            parts: list[str] = []
            for item in raw_content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(item.get("text", ""))
                elif isinstance(item, str):
                    parts.append(item)
            content = "".join(parts)
        else:
            content = raw_content or ""
        return GeneratorResponse(
            content=content,
            model=data.get("model", ""),
            usage=data.get("usage", {}),
            raw=data,
            # OpenAI-style providers return ``reasoning_content``; vLLM's
            # qwen3 reasoning parser returns ``reasoning`` instead.
            reasoning_content=str(
                message.get("reasoning_content") or message.get("reasoning") or ""
            ),
            finish_reason=str(choice.get("finish_reason") or ""),
        )


class VertexGeminiClient:
    """Async Vertex AI Gemini client with the same generate API."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._semaphore = asyncio.Semaphore(config.batch_size)
        self._client = None
        self._client_lock = asyncio.Lock()
        self._trace_recorder = None

    def set_trace_recorder(self, recorder) -> None:
        self._trace_recorder = recorder

    async def generate(
        self,
        messages: list[Message] | ProviderConversationRequest | None = None,
        schema: dict | None = None,
        trace_context: dict[str, Any] | None = None,
        *,
        request: ProviderConversationRequest | None = None,
    ) -> GeneratorResponse:
        if isinstance(messages, ProviderConversationRequest):
            if request is not None:
                raise TypeError("conversation request was supplied twice")
            request = messages
            messages = None
        if request is not None:
            wire_messages = request.contents
        elif messages is not None:
            wire_messages = messages
        else:
            raise TypeError("generate requires messages or a conversation request")
        async with self._semaphore:
            started = monotonic()
            trace_state: dict[str, Any] = {"attempt_count": 0, "retry_events": []}
            try:
                response = await self._call_with_retry(
                    wire_messages, schema, request=request, trace_state=trace_state
                )
            except Exception as exc:
                if self._trace_recorder is not None:
                    await self._trace_recorder.record_failure(
                        provider=self.config.provider,
                        model=self.config.model,
                        messages=wire_messages,
                        schema=schema,
                        error=exc,
                        duration_ms=int((monotonic() - started) * 1000),
                        trace_context=trace_context,
                        attempt_count=trace_state["attempt_count"],
                        retry_events=trace_state["retry_events"],
                    )
                raise
            if self._trace_recorder is not None:
                await self._trace_recorder.record_success(
                    provider=self.config.provider,
                    model=self.config.model,
                    messages=wire_messages,
                    schema=schema,
                    response=response,
                    duration_ms=int((monotonic() - started) * 1000),
                    trace_context=trace_context,
                    attempt_count=trace_state["attempt_count"],
                    retry_events=trace_state["retry_events"],
                )
            return response

    def _get_sdk(self):
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:  # pragma: no cover
            raise GeneratorError(
                "google-genai is not installed. Install it with `pip install google-genai` "
                "to use a Gemini provider."
            ) from exc
        return genai, types

    @property
    def _request_label(self) -> str:
        return "Vertex"

    def _build_client(self):
        genai, _ = self._get_sdk()
        api_key = self.config.resolve_api_key()
        if not api_key:
            source = self.config.api_key_env or "configured api_key"
            raise GeneratorError(
                f"Missing Vertex API key for model '{self.config.model}'. Expected a value from {source}."
            )
        project = self.config.resolve_vertex_project()
        location = self.config.resolve_vertex_location() or "asia-southeast1"
        if project:
            os.environ.setdefault("GOOGLE_CLOUD_PROJECT", project)
        if location:
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", location)
        timeout_ms = max(1000, int(float(self.config.request_timeout_seconds) * 1000))
        http_options = {
            "api_version": "v1",
            "timeout": timeout_ms,
            "retry_options": {"attempts": max(1, min(2, int(self.config.request_retry_limit)))},
        }
        return genai.Client(
            vertexai=True,
            api_key=api_key,
            http_options=http_options,
        )

    async def _reset_client(self) -> None:
        async with self._client_lock:
            client = self._client
            self._client = None
        if client is None:
            return
        for method_name in ("close",):
            method = getattr(client, method_name, None)
            if callable(method):
                try:
                    await asyncio.to_thread(method)
                except Exception:
                    pass
                break

    @staticmethod
    def _split_messages(messages: list[Message]) -> tuple[str, str]:
        system_parts: list[str] = []
        user_parts: list[str] = []
        for message in messages:
            role = str(message.get("role") or "")
            content = str(message.get("content") or "")
            if not content:
                continue
            if role == "system":
                system_parts.append(content)
            else:
                user_parts.append(f"{role.upper()}:\n{content}")
        return "\n\n".join(system_parts).strip(), "\n\n".join(user_parts).strip()

    async def _call_once(
        self,
        messages: list[Message],
        schema: dict | None,
        request: ProviderConversationRequest | None = None,
    ) -> GeneratorResponse:
        _, types = self._get_sdk()
        client = self._build_client()
        if request is not None:
            system_instruction = request.system_instruction
            contents = deepcopy(request.contents)
        else:
            system_instruction, contents = self._split_messages(messages)
        if not contents:
            raise GeneratorError("Gemini generation requires at least one non-system message.")

        config_kwargs: dict[str, Any] = {
            "max_output_tokens": self.config.max_tokens,
        }
        if self.config.labels:
            config_kwargs["labels"] = self.config.labels
        if self.config.temperature is not None:
            config_kwargs["temperature"] = self.config.temperature
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if self.config.top_p is not None:
            config_kwargs["top_p"] = self.config.top_p
        if self.config.thinking_tokens is not None:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.config.thinking_tokens
            )
        elif self.config.effort is not None:
            thinking_levels = {
                "minimal": types.ThinkingLevel.MINIMAL,
                "low": types.ThinkingLevel.LOW,
                "medium": types.ThinkingLevel.MEDIUM,
                "high": types.ThinkingLevel.HIGH,
            }
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=thinking_levels[self.config.effort],
                include_thoughts=True,
            )
        if self.config.output_format in {"json_object", "json_schema"}:
            config_kwargs["response_mime_type"] = "application/json"
            if self.config.output_format == "json_schema" and schema is not None:
                config_kwargs["response_schema"] = schema
        else:
            config_kwargs["response_mime_type"] = "text/plain"

        for key, value in self.config.extra_body.items():
            if key == "thinking":
                if isinstance(value, dict) and str(value.get("type") or "").lower() == "disabled":
                    config_kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
                continue
            config_kwargs[key] = value

        timeout_seconds = max(1.0, float(self.config.request_timeout_seconds))
        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.models.generate_content,
                    model=self.config.model,
                    contents=contents,
                    config=types.GenerateContentConfig(**config_kwargs),
                ),
                timeout=timeout_seconds + 5.0,
            )
        finally:
            close = getattr(client, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        text = (getattr(response, "text", None) or "").strip()
        reasoning_parts: list[str] = []
        reasoning_signature_present = False
        candidates = list(getattr(response, "candidates", None) or [])
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            for part in list(getattr(content, "parts", None) or []):
                if getattr(part, "thought_signature", None):
                    reasoning_signature_present = True
                if bool(getattr(part, "thought", False)):
                    thought_text = str(getattr(part, "text", None) or "").strip()
                    if thought_text:
                        reasoning_parts.append(thought_text)
        usage = {}
        usage_meta = getattr(response, "usage_metadata", None)
        if usage_meta is not None:
            for attr, key in (
                ("prompt_token_count", "prompt_tokens"),
                ("candidates_token_count", "completion_tokens"),
                ("thoughts_token_count", "reasoning_tokens"),
                ("total_token_count", "total_tokens"),
            ):
                value = getattr(usage_meta, attr, None)
                if isinstance(value, int):
                    usage[key] = value
        raw = response.to_json_dict() if hasattr(response, "to_json_dict") else {"text": text}
        if not isinstance(raw, dict):
            raw = {"text": text}
        candidates_raw = raw.get("candidates") or []
        assistant_content = None
        if isinstance(candidates_raw, list) and candidates_raw:
            assistant_content = candidates_raw[0].get("content")
        return GeneratorResponse(
            content=text,
            model=self.config.model,
            usage=usage,
            raw=raw,
            reasoning_content="\n\n".join(reasoning_parts),
            reasoning_signature_present=reasoning_signature_present,
            finish_reason=str(getattr(candidates[0], "finish_reason", "") or "") if candidates else "",
            assistant_content=assistant_content,
        )

    async def _call_with_retry(
        self,
        messages: list[Message],
        schema: dict | None,
        request: ProviderConversationRequest | None = None,
        trace_state: dict[str, Any] | None = None,
    ) -> GeneratorResponse:
        max_retries = max(1, int(self.config.request_retry_limit))
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            if trace_state is not None:
                trace_state["attempt_count"] = attempt
            try:
                return await self._call_once(messages, schema, request=request)
            except GeneratorError:
                raise
            except asyncio.TimeoutError as exc:
                last_error = exc
                if trace_state is not None:
                    trace_state["retry_events"].append({"attempt": attempt, "event": "timeout"})
                await self._reset_client()
                if attempt >= max_retries:
                    raise GeneratorError(
                        f"{self._request_label} request timed out after "
                        f"{max_retries} attempts"
                    ) from exc
                print(
                    f"{self._request_label} request timed out "
                    f"(attempt {attempt}/{max_retries}), retrying...",
                    flush=True,
                )
            except Exception as exc:
                last_error = exc
                if trace_state is not None:
                    trace_state["retry_events"].append(
                        {
                            "attempt": attempt,
                            "event": "provider_error",
                            "error_type": exc.__class__.__name__,
                            "message": str(exc),
                        }
                    )
                await self._reset_client()
                if attempt >= max_retries:
                    raise GeneratorError(
                        f"{self._request_label} request failed after "
                        f"{max_retries} attempts: "
                        f"{exc.__class__.__name__}: {exc}"
                    ) from exc
                print(
                    f"{self._request_label} request failed "
                    f"(attempt {attempt}/{max_retries}): "
                    f"{exc.__class__.__name__}: {exc}",
                    flush=True,
                )
            await asyncio.sleep(
                self.config.request_retry_base_delay_seconds * (2 ** (attempt - 1))
            )
        raise GeneratorError(
            f"{self._request_label} request failed after "
            f"{max_retries} attempts: {last_error}"
        )


class GoogleGeminiClient(VertexGeminiClient):
    """Async Gemini Developer API client authenticated with an API key."""

    @property
    def _request_label(self) -> str:
        return "Gemini Developer API"

    def _build_client(self):
        genai, _ = self._get_sdk()
        api_key = self.config.resolve_api_key()
        if not api_key:
            source = self.config.api_key_env or "configured api_key"
            raise GeneratorError(
                f"Missing Gemini Developer API key for model '{self.config.model}'. "
                f"Expected a value from {source}."
            )
        timeout_ms = max(1000, int(float(self.config.request_timeout_seconds) * 1000))
        http_options = {
            "api_version": "v1beta",
            "timeout": timeout_ms,
            "retry_options": {
                "attempts": max(1, min(2, int(self.config.request_retry_limit)))
            },
        }
        return genai.Client(
            api_key=api_key,
            http_options=http_options,
        )
