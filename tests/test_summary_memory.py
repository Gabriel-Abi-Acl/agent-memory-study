"""Contrato: o sumarizador só é chamado quando o buffer literal estoura o limite."""

from agent_memory_study.memory.summary import FakeSummarizer, SummaryMemory
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
