"""Gerador de respostas. A implementação falsa não chama rede.

`FakeLLM` devolve um texto fixo e guarda o prompt. Não simula acerto
de gabarito: correção da resposta é métrica do PR3, com modelo real no PR4.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from agent_memory_study.types import Message


class LLM(Protocol):
    def complete(self, messages: Sequence[Message]) -> str:
        """Gera a fala do assistente a partir do contexto já montado."""


class FakeLLM:
    def __init__(self, reply: str = "resposta-fake") -> None:
        self.reply = reply
        self.prompts: list[tuple[Message, ...]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.prompts.append(tuple(messages))
        return self.reply
