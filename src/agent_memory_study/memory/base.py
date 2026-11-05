"""Contrato comum das três estratégias.

O agente lê o contexto e só depois grava o turno. A pergunta atual não
precisa já estar indexada para aparecer no prompt.
"""

from __future__ import annotations

from typing import Protocol

from agent_memory_study.types import Message


class Memory(Protocol):
    def add(self, message: Message) -> None:
        """Guarda uma mensagem que já aconteceu."""

    def context(self, query: str | None = None) -> list[Message]:
        """Devolve o trecho que esta estratégia mostra ao modelo."""
