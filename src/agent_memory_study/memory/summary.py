"""Resumo acumulado mais um buffer curto de mensagens literais.

Quando o buffer passa de `recent_limit`, o excedente sai do literal e
entra no sumarizador. Esse objeto é o hook: dá para trocar o algoritmo
sem mudar a estratégia.
"""

from __future__ import annotations

from typing import Protocol

from agent_memory_study.types import Message

_SUMMARY_PREFIX = "Resumo da conversa até aqui: "


class Summarizer(Protocol):
    def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        """Condensa `messages` sobre o resumo anterior e devolve o texto novo."""


class FakeSummarizer:
    """Concatena em vez de resumir, e registra cada chamada.

    O registro é o que o teste do hook observa. O texto resultante não é
    um resumo de modelo e não serve como dado de experimento.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Message, ...]]] = []

    def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        folded = tuple(messages)
        self.calls.append((previous_summary, folded))
        chunk = " ".join(message.content for message in folded)
        if previous_summary:
            return f"{previous_summary} {chunk}"
        return chunk


class SummaryMemory:
    def __init__(self, summarizer: Summarizer, recent_limit: int) -> None:
        if recent_limit < 1:
            raise ValueError("recent_limit must be >= 1")
        self.summarizer = summarizer
        self.recent_limit = recent_limit
        self.summary = ""
        self._recent: list[Message] = []

    def add(self, message: Message) -> None:
        self._recent.append(message)
        overflow = len(self._recent) - self.recent_limit
        if overflow <= 0:
            return
        folded = self._recent[:overflow]
        self._recent = self._recent[overflow:]
        self.summary = self.summarizer.summarize(self.summary, folded)

    def context(self, query: str | None = None) -> list[Message]:
        if not self.summary:
            return list(self._recent)
        header = Message(role="system", content=f"{_SUMMARY_PREFIX}{self.summary}")
        return [header, *self._recent]
