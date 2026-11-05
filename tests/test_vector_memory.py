"""Contrato: a consulta lexical recupera a mensagem que contém o fato plantado."""

import pytest

from agent_memory_study.dialogues import load_dialogues
from agent_memory_study.embeddings import FakeEmbedder
from agent_memory_study.memory.vector import VectorMemory


@pytest.mark.parametrize("dialogue_id", ["pref-cor", "codigo-projeto", "gato-milo"])
def test_vector_memory_recalls_planted_fact_from_placeholder_dialogue(dialogue_id: str) -> None:
    dialogue = next(item for item in load_dialogues() if item.id == dialogue_id)
    probe = dialogue.probes[0]
    fact = next(item for item in dialogue.facts if item.id == probe.expected_fact_ids[0])
    history = dialogue.messages[: probe.after_turn + 1]
    memory = VectorMemory(FakeEmbedder(), top_k=1)

    for message in history:
        memory.add(message)

    retrieved = memory.context(probe.question)

    assert len(retrieved) == 1
    assert fact.text in retrieved[0].content
    assert retrieved[0] == history[fact.planted_at_turn]
