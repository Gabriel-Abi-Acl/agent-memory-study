"""Configuração lida do ambiente. Defaults são offline.

O processo não carrega `.env` sozinho e não grava chave em lugar nenhum.
Provedor diferente de `fake` falha: neste PR não há cliente de API.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_FAKE = "fake"


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
            f"embedding_model={self.embedding_model!r})"
        )


def load_settings(environ: Mapping[str, str] | None = None) -> Settings:
    env = os.environ if environ is None else environ
    api_key = env.get("OPENAI_API_KEY", "").strip()
    return Settings(
        llm_provider=_provider(env, "LLM_PROVIDER"),
        embedding_provider=_provider(env, "EMBEDDING_PROVIDER"),
        window_size=_positive_int(env, "WINDOW_SIZE", 8),
        summary_recent_limit=_positive_int(env, "SUMMARY_RECENT_LIMIT", 4),
        vector_top_k=_positive_int(env, "VECTOR_TOP_K", 3),
        openai_api_key=api_key or None,
        openai_model=env.get("OPENAI_MODEL", "").strip() or "gpt-4o-mini",
        embedding_model=env.get("EMBEDDING_MODEL", "").strip() or "text-embedding-3-small",
    )


def _provider(env: Mapping[str, str], name: str) -> str:
    value = env.get(name, _FAKE).strip() or _FAKE
    if value != _FAKE:
        raise ValueError(f"{name}={value!r} is not implemented; this PR only accepts {_FAKE!r}")
    return value


def _positive_int(env: Mapping[str, str], name: str, default: int) -> int:
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value
