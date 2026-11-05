"""Janela deslizante: só as últimas N mensagens ficam no contexto.

Mensagem que sai da janela deixa de ser visível. Esse esquecimento é a
falha que o estudo quer observar, não algo a mascarar aqui.
"""

from __future__ import annotations

from agent_memory_study.types import Message


class WindowMemory:
    def __init__(self, size: int) -> None:
        if size < 1:
            raise ValueError("window size must be >= 1")
        self.size = size
        self._messages: list[Message] = []

    def add(self, message: Message) -> None:
        self._messages.append(message)
        overflow = len(self._messages) - self.size
        if overflow > 0:
            del self._messages[:overflow]

    def context(self, query: str | None = None) -> list[Message]:
        return list(self._messages)
