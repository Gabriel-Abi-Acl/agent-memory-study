"""Configuração lida do ambiente. O default é offline.

O processo não carrega `.env` sozinho e não grava chave. `openai` só chama
a rede quando o cliente do harness é construído com `OPENAI_API_KEY`.
"""

from __future__ import annotations

import os
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass

_PROVIDERS = frozenset({"fake", "openai"})
_DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass(frozen=True)
class Settings:
    llm_provider: str
    embedding_provider: str
    window_size: int
    summary_recent_limit: int
    vector_top_k: int
    openai_api_key: str | None
    openai_model: str
    embedding_model: str
    openai_base_url: str
    llm_max_attempts: int
    llm_retry_base_delay_s: float
    llm_retry_max_delay_s: float
    llm_timeout_s: float
    prompt_usd_per_million: float | None
    completion_usd_per_million: float | None

    def __repr__(self) -> str:
        # A chave não entra no repr: log de debug não pode vazar segredo.
        key_state = "set" if self.openai_api_key else "empty"
        return (
            "Settings("
            f"llm_provider={self.llm_provider!r}, "
            f"embedding_provider={self.embedding_provider!r}, "
            f"window_size={self.window_size}, "
            f"summary_recent_limit={self.summary_recent_limit}, "
            f"vector_top_k={self.vector_top_k}, "
            f"openai_api_key=<{key_state}>, "
            f"openai_model={self.openai_model!r}, "
            f"embedding_model={self.embedding_model!r}, "
            f"llm_max_attempts={self.llm_max_attempts})"
        )


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if environ is None else environ
    api_key = env.get("OPENAI_API_KEY", "").strip()
    prompt_price = _optional_float(env, "PROMPT_USD_PER_MILLION")
    completion_price = _optional_float(env, "COMPLETION_USD_PER_MILLION")
    if (prompt_price is None) != (completion_price is None):
        raise ValueError(
            "PROMPT_USD_PER_MILLION and COMPLETION_USD_PER_MILLION must be set together"
        )
    return Settings(
        llm_provider=_provider(env, "LLM_PROVIDER"),
        embedding_provider=_provider(env, "EMBEDDING_PROVIDER"),
        window_size=_positive_int(env, "WINDOW_SIZE", 8),
        summary_recent_limit=_positive_int(env, "SUMMARY_RECENT_LIMIT", 4),
        vector_top_k=_positive_int(env, "VECTOR_TOP_K", 3),
        openai_api_key=api_key or None,
        openai_model=env.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini",
        embedding_model=env.get("EMBEDDING_MODEL", "").strip() or "text-embedding-3-small",
        openai_base_url=_base_url(env),
        llm_max_attempts=_positive_int(env, "LLM_MAX_ATTEMPTS", 4),
        llm_retry_base_delay_s=_non_negative_float(env, "LLM_RETRY_BASE_DELAY_S", 0.5),
        llm_retry_max_delay_s=_non_negative_float(env, "LLM_RETRY_MAX_DELAY_S", 8.0),
        llm_timeout_s=_positive_float(env, "LLM_TIMEOUT_S", 30.0),
        prompt_usd_per_million=prompt_price,
        completion_usd_per_million=completion_price,
    )


def _provider(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, "fake").strip() or "fake"
    if value not in _PROVIDERS:
        allowed = ", ".join(sorted(_PROVIDERS))
        raise ValueError(f"{name}={value!r} is not implemented; expected one of: {allowed}")
    return value


def _base_url(env: Mapping[str, str]) -> str:
    raw = env.get("OPENAI_BASE_URL", "").strip() or _DEFAULT_BASE_URL
    value = raw.rstrip("/")
    parsed = urllib.parse.urlparse(value)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.netloc
        or parsed.username
        or parsed.password
    ):
        raise ValueError("OPENAI_BASE_URL must be an http(s) URL without credentials")
    return value


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    return _parse_int(env, name, default, minimum=1)


def _parse_int(env: Mapping[str, str], name: str, default: int, *, minimum: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _optional_float(env: Mapping[str, str], name: str) -> float | None:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return None
    return _parse_float(name, raw, minimum=0.0)


def _non_negative_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    return _parse_float(name, raw, minimum=0.0)


def _positive_float(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    value = _parse_float(name, raw, minimum=0.0)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _parse_float(name: str, raw: str, *, minimum: float) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value
