"""Contrato: o sumarizador só é chamado quando o buffer literal estoura o limite."""

import pytest

from agent_memory_study.memory.summary import FakeSummarizer, LLMSummarizer, SummaryMemory
from agent_memory_study.types import Message


def test_summary_update_hook_folds_overflow_and_receives_previous_summary() -> None:
    summarizer = FakeSummarizer()
    memory = SummaryMemory(summarizer, recent_limit=2)
    planted = Message("user", "O gato se chama Milo.")
    acknowledgement = Message("assistant", "Anotado.")
    later = Message("user", "O almoço foi lasanha.")
    closing = Message("assistant", "Certo.")

    memory.add(planted)
    memory.add(acknowledgement)
    assert summarizer.calls == []

    memory.add(later)

    assert summarizer.calls == [("", (planted,))]
    assert memory.summary == "O gato se chama Milo."
    assert memory.context() == [
        Message("system", "Resumo da conversa até aqui: O gato se chama Milo."),
        acknowledgement,
        later,
    ]

    memory.add(closing)

    assert summarizer.calls[1] == ("O gato se chama Milo.", (acknowledgement,))
    assert "Milo" in memory.summary
    assert memory.context()[-2:] == [later, closing]


def test_llm_summarizer_forwards_the_previous_summary_and_strips_the_reply() -> None:
    class _Client:
        def __init__(self) -> None:
            self.prompts: list[list[Message]] = []

        def complete(self, messages: list[Message]) -> object:
            self.prompts.append(list(messages))
            return type("Reply", (), {"text": "  resumo novo  "})()

    client = _Client()
    text = LLMSummarizer(client).summarize("antes", [Message("user", "O código é Nimbus-42.")])
    sent = "\n".join(message.content for message in client.prompts[0])

    assert text == "resumo novo"
    assert "antes" in sent
    assert "Nimbus-42" in sent


def test_llm_summarizer_rejects_an_empty_summary() -> None:
    class _Client:
        def complete(self, messages: list[Message]) -> object:
            return type("Reply", (), {"text": "   "})()

    with pytest.raises(ValueError, match="empty"):
        LLMSummarizer(_Client()).summarize("", [Message("user", "oi")])
