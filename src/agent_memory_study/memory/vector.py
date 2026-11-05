"""Índice vetorial só em memória, sem persistência entre processos.

`context` devolve os `top_k` itens mais próximos da consulta. Com o
embedder falso a ordem é lexical; distrator semântico fica para o PR4.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent_memory_study.embeddings import Embedder
from agent_memory_study.types import Message


@dataclass
class _Stored:
    message: Message
    vector: list[float]


class VectorMemory:
    def __init__(self, embedder: Embedder, top_k: int) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self.embedder = embedder
        self.top_k = top_k
        self._items: list[_Stored] = []

    def add(self, message: Message) -> None:
        vector = _one_vector(self.embedder.embed([message.content]))
        self._items.append(_Stored(message=message, vector=vector))

    def context(self, query: str | None = None) -> list[Message]:
        if not self._items:
            return []
        if query is None:
            return [item.message for item in self._items]
        query_vector = _one_vector(self.embedder.embed([query]))
        ranked = sorted(
            enumerate(self._items),
            key=lambda pair: (_cosine(query_vector, pair[1].vector), pair[0]),
            reverse=True,
        )
        return [item.message for _, item in ranked[: self.top_k]]


def _one_vector(vectors: list[list[float]]) -> list[float]:
    if len(vectors) != 1:
        raise ValueError("embedder must return one vector per text")
    return vectors[0]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)
