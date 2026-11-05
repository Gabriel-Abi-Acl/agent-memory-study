"""Escolhe a estratégia pelo nome, com os tamanhos vindos da configuração.

Os três nomes são o fator do estudo. O laço do agente não muda entre eles.
"""

from __future__ import annotations

from agent_memory_study.config import Settings
from agent_memory_study.embeddings import Embedder, FakeEmbedder
from agent_memory_study.memory.base import Memory
from agent_memory_study.memory.summary import FakeSummarizer, Summarizer, SummaryMemory
from agent_memory_study.memory.vector import VectorMemory
from agent_memory_study.memory.window import WindowMemory

STRATEGIES = ("window", "summary", "vector")


def build_memory(
    strategy: str,
    settings: Settings,
    *,
    summarizer: Summarizer | None = None,
    embedder: Embedder | None = None,
) -> Memory:
    if strategy == "window":
        return WindowMemory(size=settings.window_size)
    if strategy == "summary":
        return SummaryMemory(
            summarizer if summarizer is not None else FakeSummarizer(),
            recent_limit=settings.summary_recent_limit,
        )
    if strategy == "vector":
        return VectorMemory(
            embedder if embedder is not None else FakeEmbedder(),
            top_k=settings.vector_top_k,
        )
    allowed = ", ".join(STRATEGIES)
    raise ValueError(f"unknown memory strategy: {strategy!r}; expected one of: {allowed}")
