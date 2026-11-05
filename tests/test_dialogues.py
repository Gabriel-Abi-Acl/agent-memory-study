"""Contrato: o fixture mínimo tem diálogos sintéticos com fato antes da sondagem."""

from agent_memory_study.dialogues import load_dialogues


def test_placeholder_fixture_loads_three_synthetic_dialogues() -> None:
    dialogues = load_dialogues()

    assert {dialogue.id for dialogue in dialogues} == {
        "pref-cor",
        "codigo-projeto",
        "gato-milo",
    }
    for dialogue in dialogues:
        assert dialogue.facts
        assert dialogue.probes
        for probe in dialogue.probes:
            for fact_id in probe.expected_fact_ids:
                fact = next(item for item in dialogue.facts if item.id == fact_id)
                assert fact.text in dialogue.messages[fact.planted_at_turn].content
                assert fact.planted_at_turn <= probe.after_turn
