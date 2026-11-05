"""Embeddings offline por hashing de tokens.

Não é modelo semântico: a proximidade é sobreposição lexical, estável
e sem rede. Vetor deste módulo não entra como resultado experimental.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

_TOKEN = re.compile(r"[0-9A-Za-zÀ-ÿ]+")
_STOPWORDS = frozenset(
    {
        "a",
        "o",
        "os",
        "as",
        "de",
        "da",
        "do",
        "das",
        "dos",
        "e",
        "é",
        "que",
        "qual",
        "quais",
        "um",
        "uma",
        "em",
        "no",
        "na",
        "nos",
        "nas",
        "para",
        "por",
        "com",
        "se",
        "eu",
        "me",
        "meu",
        "minha",
        "seu",
        "sua",
        "foi",
        "hoje",
        "ao",
        "à",
        "às",
        "aos",
    }
)


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Um vetor por texto, na mesma ordem."""


class FakeEmbedder:
    def __init__(self, dim: int = 256) -> None:
        if dim < 1:
            raise ValueError("dim must be >= 1")
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [_embed_text(text, self.dim) for text in texts]


def _tokens(text: str) -> list[str]:
    return [token for token in _TOKEN.findall(text.lower()) if token not in _STOPWORDS]


def _embed_text(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in _tokens(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]
