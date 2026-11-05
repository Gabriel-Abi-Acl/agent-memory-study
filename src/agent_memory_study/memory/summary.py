"""Resumo acumulado mais um buffer curto de mensagens literais.

Quando o buffer passa de `recent_limit`, o excedente sai do literal e
entra no sumarizador. Esse objeto é o hook: dá para trocar o algoritmo
sem mudar a estratégia.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent_memory_study.types import Message

SUMMARY_PREFIX = "Resumo da conversa até aqui: "


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
        header = Message(role="system", content=f"{SUMMARY_PREFIX}{self.summary}")
        return [header, *self._recent]


class _CompletionText(Protocol):
    text: str


class _Chat(Protocol):
    def complete(self, messages: Sequence[Message]) -> _CompletionText:
        """Devolve a fala do modelo. Retry e cache ficam no cliente."""


class LLMSummarizer:
    """Pede ao modelo o resumo novo e substitui o texto anterior.

    Literal corrigido só sobrevive se o modelo o reescrever. O cliente é quem
    aplica retry e cache; este objeto não abre conexão.
    """

    def __init__(self, client: _Chat) -> None:
        self.client = client

    def summarize(self, previous_summary: str, messages: list[Message]) -> str:
        transcript = "\n".join(f"{message.role}: {message.content}" for message in messages)
        previous = previous_summary or "(vazio)"
        completion = self.client.complete(
            [
                Message(
                    "system",
                    "Atualize o resumo da conversa em português. "
                    "Preserve valores literais (nomes, códigos, cores, números). "
                    "Se uma mensagem corrigir um valor, fique só com o valor novo. "
                    "Responda apenas com o resumo.",
                ),
                Message(
                    "user",
                    f"Resumo anterior:\n{previous}\n\nMensagens novas:\n{transcript}",
                ),
            ]
        )
        text = completion.text.strip()
        if not text:
            raise ValueError("summarizer returned an empty summary")
        return text
