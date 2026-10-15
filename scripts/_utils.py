"""Shared helpers for all scripts (no pipeline logic here)."""
from __future__ import annotations

import os
from pathlib import Path


MODEL_OVERRIDE_KEYS = {
    "model",
    "provider",
    "conversation_protocol",
    "base_url",
    "base_url_env",
    "api_key",
    "api_key_env",
    "vertex_project",
    "vertex_project_env",
    "vertex_location",
    "vertex_location_env",
    "max_tokens",
    "temperature",
    "top_p",
    "batch_size",
    "output_format",
    "extra_body",
    "labels",
    "budget",
    "thinking_tokens",
    "max_completion_tokens",
    "effort",
    "request_timeout_seconds",
    "request_retry_limit",
    "request_retry_base_delay_seconds",
    "api_mode",
    "reasoning_summary",
    "background_poll_interval_seconds",
}


def load_repo_env(search_from: Path | None = None) -> Path | None:
    """Load .env from the repository tree into os.environ if present.

    This is a minimal loader to avoid adding a runtime dependency just for local
    API-key setup. Existing environment variables take precedence.
    """
    start = (search_from or Path(__file__).resolve()).resolve()
    current = start if start.is_dir() else start.parent

    for directory in (current, *current.parents):
        env_path = directory / ".env"
        if not env_path.exists():
            continue

        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            cleaned = value.strip().strip("\"'`")
            os.environ.setdefault(key, cleaned)
        return env_path

    return None


def merge_model_config(model_cfg: dict, overrides: dict | None = None) -> dict:
    """Merge stable model config with task-specific run overrides.

    Model configs should hold stable connection/provider defaults.
    Run configs may override sampling and output parameters per task.
    """
    merged = dict(model_cfg)
    if not overrides:
        return merged

    for key in MODEL_OVERRIDE_KEYS:
        if key in overrides:
            merged[key] = overrides[key]
    return merged


def build_generator(model_cfg: dict):
    """Construct the configured provider client from a model config dict.

    The `budget:` sub-key in the YAML is flattened into GeneratorConfig so
    fields like `thinking_tokens`, `effort`, and `max_completion_tokens` are
    set directly — no separate ReasoningBudget object needed.
    """
    load_repo_env(Path(__file__).resolve().parents[1])

    from generator import (
        GeneratorConfig,
        GoogleGeminiClient,
        OpenAICompatClient,
        VertexGeminiClient,
    )
    from generator.conversation_adapters import get_conversation_adapter

    flat = dict(model_cfg)
    budget = flat.pop("budget", {}) or {}
    flat.update(budget)  # thinking_tokens / effort / max_completion_tokens now at top level
    # "default" means the provider's own reasoning behavior; send nothing.
    if flat.get("effort") == "default":
        flat["effort"] = None

    base_url_env = flat.pop("base_url_env", None)
    if base_url_env is not None:
        if not isinstance(base_url_env, str) or not base_url_env.strip():
            raise ValueError("base_url_env must name a non-empty environment variable")
        base_url_env = base_url_env.strip()
        base_url = os.environ.get(base_url_env)
        if not base_url or not base_url.strip():
            raise ValueError(
                f"Missing base URL. Expected a value from {base_url_env}."
            )
        flat["base_url"] = base_url.strip()

    # Conversation protocol is explicit configuration, never inferred from a
    # model name or endpoint. One-shot generation still uses the same client;
    # the adapter is selected by the benchmark runner after this validation.
    get_conversation_adapter(flat.get("conversation_protocol"))

    gen_cfg = GeneratorConfig(**{k: v for k, v in flat.items()
                                  if k in GeneratorConfig.__dataclass_fields__})
    if gen_cfg.provider == "vertex_gemini":
        return VertexGeminiClient(gen_cfg)
    if gen_cfg.provider == "google_gemini":
        return GoogleGeminiClient(gen_cfg)
    return OpenAICompatClient(gen_cfg)
