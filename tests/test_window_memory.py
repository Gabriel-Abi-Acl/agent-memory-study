"""Contrato: acima de `size` mensagens, a mais antiga sai e não volta ao contexto."""

from agent_memory_study.memory.window import WindowMemory
from agent_memory_study.types import Message


def test_window_evicts_oldest_message_when_size_is_exceeded() -> None:
    memory = WindowMemory(size=2)
    forgotten = Message("user", "fato antigo: azul-petróleo")
    kept_first = Message("assistant", "ainda visível")
    kept_last = Message("user", "mensagem recente")

    memory.add(forgotten)
    memory.add(kept_first)
    assert memory.context() == [forgotten, kept_first]

    memory.add(kept_last)

    assert memory.context() == [kept_first, kept_last]
    assert all("azul-petróleo" not in message.content for message in memory.context())
