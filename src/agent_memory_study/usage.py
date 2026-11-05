"""Tokens e custo estimado das chamadas do harness.

Sem o campo `usage` da API a contagem é teto(caracteres / 4). O preço é a
tabela de lista ou o par de variáveis de ambiente, nunca uma fatura medida.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from agent_memory_study.config import Settings
from agent_memory_study.types import Message

# USD por 1 milhão de tokens (entrada, saída). Lista pública usada só na estimativa.
_LIST_PRICES: dict[str, tuple[float, float]] = {
    "fake": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.5, 10.0),
    "text-embedding-3-small": (0.02, 0.0),
}


@dataclass(frozen=True)
class Completion:
    """Resposta já com tokens e custo desta execução.

    Hit de cache mantém os tokens e zera o custo: a execução não pagou de novo.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    estimated_cost_usd: float
    cached: bool
    model: str


def estimate_tokens(text: str) -> int:
    """Teto de caracteres/4. Vazio vale zero para não inflar chamada sem texto."""

    if not text:
        return 0
    return (len(text) + 3) // 4


def estimate_message_tokens(messages: Sequence[Message]) -> int:
    return sum(estimate_tokens(message.content) for message in messages)


def estimate_cost_usd(
    settings: Settings,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    *,
    cached: bool,
) -> float:
    """Custo desta chamada. Cache hit é zero mesmo com tokens registrados."""

    if cached:
        return 0.0
    prompt_price, completion_price = _prices(settings, model)
    return (prompt_tokens * prompt_price + completion_tokens * completion_price) / 1_000_000


def _prices(settings: Settings, model: str) -> tuple[float, float]:
    # Override vale para o modelo de chat configurado e para modelo fora da tabela.
    # Embedding com preço de lista não herda o preço do chat.
    override = _price_override(settings)
    if override is not None and model == settings.openai_model:
        return override
    listed = _LIST_PRICES.get(model)
    if listed is not None:
        return listed
    if override is not None:
        return override
    return (0.0, 0.0)


def _price_override(settings: Settings) -> tuple[float, float] | None:
    prompt = settings.prompt_usd_per_million
    completion = settings.completion_usd_per_million
    if prompt is None and completion is None:
        return None
    if prompt is None or completion is None:
        raise ValueError(
            "PROMPT_USD_PER_MILLION and COMPLETION_USD_PER_MILLION must be set together"
        )
    return (prompt, completion)
