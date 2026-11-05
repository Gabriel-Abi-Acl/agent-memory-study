"""Mensagens e o formato mínimo de um diálogo sintético.

O gabarito da resposta (texto esperado) entra no PR2. Aqui o fato plantado
é só um trecho literal da mensagem em que ele aparece.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Role = Literal["user", "assistant", "system"]


@dataclass(frozen=True)
class Message:
    role: Role
    content: str


@dataclass(frozen=True)
class PlantedFact:
    """Trecho que uma sondagem posterior precisa achar no contexto.

    `text` é subcadeia da mensagem de índice `planted_at_turn`.
    """

    id: str
    text: str
    planted_at_turn: int


@dataclass(frozen=True)
class Probe:
    """Pergunta feita com o histórico já gravado até `after_turn`, inclusive."""

    id: str
    question: str
    after_turn: int
    expected_fact_ids: tuple[str, ...]


@dataclass(frozen=True)
class Dialogue:
    id: str
    messages: tuple[Message, ...]
    facts: tuple[PlantedFact, ...]
    probes: tuple[Probe, ...]
