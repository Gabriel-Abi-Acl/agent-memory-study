"""Contrato do dataset v1: ids únicos, gabarito, fato anterior à sondagem e os cinco tipos.

A suíte lê só o JSONL local. Não há chamada de modelo nem de rede.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from agent_memory_study.eval_dialogues import (
    DELAYED_MIN_GAP,
    IMMEDIATE_MAX_GAP,
    PROBE_TYPES,
    EvalDialogue,
    default_eval_dialogues_path,
    load_eval_dialogues,
)

_FILLERS = (
    ("assistant", "Registro feito."),
    ("user", "O almoço de hoje foi lasanha."),
    ("assistant", "Registro o almoço."),
    ("user", "A impressora do corredor está sem papel."),
    ("assistant", "Impressora anotada."),
    ("user", "O café da copa acabou."),
    ("assistant", "Café anotado."),
)


# O arquivo versionado é o contrato: estes testes lêem o JSONL do repositório.
@pytest.fixture(scope="module")
def dialogues() -> tuple[EvalDialogue, ...]:
    return load_eval_dialogues()


def test_shipped_dataset_is_a_local_versioned_jsonl(
    dialogues: tuple[EvalDialogue, ...],
) -> None:
    path = default_eval_dialogues_path()

    assert path.is_file()
    assert path.parts[-2:] == ("eval", "dialogues.v1.jsonl")
    assert 20 <= len(dialogues) <= 30
    assert all(len(dialogue.turns) >= 3 for dialogue in dialogues)


def test_ids_are_unique(dialogues: tuple[EvalDialogue, ...]) -> None:
    dialogue_ids = [dialogue.id for dialogue in dialogues]

    assert len(dialogue_ids) == len(set(dialogue_ids))
    for dialogue in dialogues:
        fact_ids = [fact.id for fact in dialogue.planted_facts]
        probe_ids = [probe.id for probe in dialogue.probes]
        assert len(fact_ids) == len(set(fact_ids))
        assert len(probe_ids) == len(set(probe_ids))


def test_every_probe_has_a_reference_answer(dialogues: tuple[EvalDialogue, ...]) -> None:
    for dialogue in dialogues:
        assert dialogue.probes
        for probe in dialogue.probes:
            assert probe.reference_answer.strip()


def test_planted_facts_appear_in_an_earlier_turn(
    dialogues: tuple[EvalDialogue, ...],
) -> None:
    for dialogue in dialogues:
        by_id = {fact.id: fact for fact in dialogue.planted_facts}
        for fact in dialogue.planted_facts:
            assert fact.text in dialogue.turns[fact.turn].content
            for index, turn in enumerate(dialogue.turns):
                if index != fact.turn:
                    assert fact.text not in turn.content
        for probe in dialogue.probes:
            probe_turn = _probe_turn(dialogue, probe.question)
            for fact_id in probe.relevant_fact_ids:
                assert by_id[fact_id].turn < probe_turn


def test_probes_come_after_the_facts_they_cite(
    dialogues: tuple[EvalDialogue, ...],
) -> None:
    for dialogue in dialogues:
        by_id = {fact.id: fact for fact in dialogue.planted_facts}
        for probe in dialogue.probes:
            probe_turn = _probe_turn(dialogue, probe.question)
            assert probe.relevant_fact_ids
            for fact_id in probe.relevant_fact_ids:
                fact = by_id[fact_id]
                assert fact.kind == "target"
                assert fact.turn < probe_turn
                assert fact.text in dialogue.turns[fact.turn].content
                assert fact.text in probe.reference_answer


def test_probe_type_distribution(dialogues: tuple[EvalDialogue, ...]) -> None:
    counts = Counter(probe.type for dialogue in dialogues for probe in dialogue.probes)

    assert counts == dict.fromkeys(PROBE_TYPES, 5)


def test_recall_gaps_follow_the_probe_type(dialogues: tuple[EvalDialogue, ...]) -> None:
    for dialogue in dialogues:
        by_id = {fact.id: fact for fact in dialogue.planted_facts}
        for probe in dialogue.probes:
            if probe.type not in {"immediate_recall", "delayed_recall"}:
                continue
            fact = by_id[probe.relevant_fact_ids[0]]
            gap = _probe_turn(dialogue, probe.question) - fact.turn
            if probe.type == "immediate_recall":
                assert 1 <= gap <= IMMEDIATE_MAX_GAP
            else:
                assert gap >= DELAYED_MIN_GAP


def test_fact_update_reference_drops_the_replaced_literal(
    dialogues: tuple[EvalDialogue, ...],
) -> None:
    seen = 0
    for dialogue in dialogues:
        by_id = {fact.id: fact for fact in dialogue.planted_facts}
        for probe in dialogue.probes:
            if probe.type != "fact_update":
                continue
            seen += 1
            current = by_id[probe.relevant_fact_ids[0]]
            assert current.supersedes is not None
            previous = by_id[current.supersedes]
            assert previous.turn < current.turn < _probe_turn(dialogue, probe.question)
            assert current.text in probe.reference_answer
            assert previous.text not in probe.reference_answer
            assert previous.text in dialogue.turns[previous.turn].content
    assert seen == 5


def test_multi_hop_joins_two_earlier_turns(dialogues: tuple[EvalDialogue, ...]) -> None:
    seen = 0
    for dialogue in dialogues:
        by_id = {fact.id: fact for fact in dialogue.planted_facts}
        for probe in dialogue.probes:
            if probe.type != "multi_hop":
                continue
            seen += 1
            assert len(probe.relevant_fact_ids) == 2
            first = by_id[probe.relevant_fact_ids[0]]
            second = by_id[probe.relevant_fact_ids[1]]
            probe_turn = _probe_turn(dialogue, probe.question)
            assert first.turn != second.turn
            assert first.turn < probe_turn
            assert second.turn < probe_turn
            assert first.text in probe.reference_answer
            assert second.text in probe.reference_answer
    assert seen == 5


def test_distractor_turns_do_not_replace_the_planted_fact(
    dialogues: tuple[EvalDialogue, ...],
) -> None:
    seen = 0
    for dialogue in dialogues:
        by_id = {fact.id: fact for fact in dialogue.planted_facts}
        for probe in dialogue.probes:
            if probe.type != "distractor":
                continue
            seen += 1
            target = by_id[probe.relevant_fact_ids[0]]
            probe_turn = _probe_turn(dialogue, probe.question)
            distractors = [
                fact
                for fact in dialogue.planted_facts
                if fact.kind == "distractor" and target.turn < fact.turn < probe_turn
            ]
            assert distractors
            assert target.text in probe.reference_answer
            for distractor in distractors:
                assert distractor.text in dialogue.turns[distractor.turn].content
                assert distractor.text not in probe.reference_answer
    assert seen == 5


def test_loader_rejects_duplicate_dialogue_ids(tmp_path: Path) -> None:
    path = _write(
        tmp_path, _recall(IMMEDIATE_MAX_GAP, "immediate_recall"), _recall(2, "immediate_recall")
    )

    with pytest.raises(ValueError, match="duplicate dialogue id"):
        load_eval_dialogues(path)


def test_loader_rejects_probe_without_reference_answer(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "immediate_recall")
    dialogue["probes"][0]["reference_answer"] = "   "

    with pytest.raises(ValueError, match="reference_answer must be a non-empty string"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_fact_missing_from_its_turn(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "immediate_recall")
    dialogue["planted_facts"][0]["text"] = "vermelho-carmim"

    with pytest.raises(ValueError, match="is not in turn"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_fact_repeated_in_a_later_turn(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "immediate_recall")
    dialogue["turns"][1]["content"] = "Registro feito: azul-petróleo."

    with pytest.raises(ValueError, match="also appears in turn"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_probe_that_does_not_come_after_its_fact(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "immediate_recall")
    dialogue["turns"] = [
        {"role": "user", "content": "Qual é a cor favorita da usuária?"},
        {"role": "assistant", "content": "Registro feito."},
        {"role": "user", "content": "A cor favorita da usuária é azul-petróleo."},
    ]
    dialogue["planted_facts"][0]["turn"] = 2

    with pytest.raises(ValueError, match="does not come after fact"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_unknown_probe_type(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "immediate_recall")
    dialogue["probes"][0]["type"] = "free_form"

    with pytest.raises(ValueError, match="invalid type"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_reference_that_misses_the_cited_fact(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "immediate_recall")
    dialogue["probes"][0]["reference_answer"] = "não informado"

    with pytest.raises(ValueError, match="reference misses fact"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_immediate_recall_beyond_the_max_gap(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP + 1, "immediate_recall")

    with pytest.raises(ValueError, match="too far"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_delayed_recall_closer_than_the_minimum_gap(tmp_path: Path) -> None:
    dialogue = _recall(DELAYED_MIN_GAP - 1, "delayed_recall")

    with pytest.raises(ValueError, match="too close"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_accepts_recall_on_the_gap_boundary(tmp_path: Path) -> None:
    immediate = load_eval_dialogues(
        _write(tmp_path, _recall(IMMEDIATE_MAX_GAP, "immediate_recall"))
    )
    delayed_path = tmp_path / "delayed.jsonl"
    _write_at(delayed_path, _recall(DELAYED_MIN_GAP, "delayed_recall"))
    delayed = load_eval_dialogues(delayed_path)

    assert immediate[0].probes[0].type == "immediate_recall"
    assert delayed[0].probes[0].type == "delayed_recall"


def test_loader_rejects_fact_update_without_supersedes(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "fact_update")

    with pytest.raises(ValueError, match="supersedes another"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_fact_update_that_keeps_the_old_literal(tmp_path: Path) -> None:
    dialogue = _correction()
    dialogue["probes"][0]["reference_answer"] = "A cor favorita atual é ocre, antes azul-petróleo."

    with pytest.raises(ValueError, match="still contains the replaced fact"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_multi_hop_with_one_fact(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "multi_hop")

    with pytest.raises(ValueError, match="must cite two facts"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_distractor_probe_without_a_distractor_turn(tmp_path: Path) -> None:
    dialogue = _recall(IMMEDIATE_MAX_GAP, "distractor")

    with pytest.raises(ValueError, match="no distractor fact"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_distractor_reference_that_adopts_the_distractor(
    tmp_path: Path,
) -> None:
    dialogue = _distractor()
    dialogue["probes"][0]["reference_answer"] = "A cor da mochila é azul-petróleo, não verde-limão."

    with pytest.raises(ValueError, match="contains distractor fact"):
        load_eval_dialogues(_write(tmp_path, dialogue))


def test_loader_rejects_a_line_that_is_not_json(tmp_path: Path) -> None:
    path = tmp_path / "dialogues.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="not valid JSON"):
        load_eval_dialogues(path)


def test_loader_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "dialogues.jsonl"
    path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        load_eval_dialogues(path)


def _probe_turn(dialogue: EvalDialogue, question: str) -> int:
    matches = [
        index
        for index, turn in enumerate(dialogue.turns)
        if turn.role == "user" and turn.content == question
    ]
    assert len(matches) == 1
    return matches[0]


def _recall(gap: int, probe_type: str) -> dict[str, Any]:
    plant = "A cor favorita da usuária é azul-petróleo."
    question = "Qual é a cor favorita da usuária?"
    needed = gap - 1
    assert 0 < needed <= len(_FILLERS)
    turns: list[dict[str, str]] = [{"role": "user", "content": plant}]
    turns.extend({"role": role, "content": content} for role, content in _FILLERS[:needed])
    turns.append({"role": "user", "content": question})
    return {
        "id": "dlg-gap",
        "turns": turns,
        "planted_facts": [
            {"id": "cor", "text": "azul-petróleo", "turn": 0, "kind": "target"},
        ],
        "probes": [
            {
                "id": "ask",
                "question": question,
                "reference_answer": "azul-petróleo",
                "relevant_fact_ids": ["cor"],
                "type": probe_type,
            }
        ],
    }


def _correction() -> dict[str, Any]:
    return {
        "id": "dlg-correcao",
        "turns": [
            {"role": "user", "content": "A cor favorita da usuária é azul-petróleo."},
            {"role": "assistant", "content": "Anotado."},
            {
                "role": "user",
                "content": "Correção: a cor favorita da usuária passou a ser ocre.",
            },
            {"role": "assistant", "content": "Correção anotada."},
            {"role": "user", "content": "Qual é a cor favorita atual da usuária?"},
        ],
        "planted_facts": [
            {"id": "cor-antiga", "text": "azul-petróleo", "turn": 0, "kind": "target"},
            {
                "id": "cor-nova",
                "text": "ocre",
                "turn": 2,
                "kind": "target",
                "supersedes": "cor-antiga",
            },
        ],
        "probes": [
            {
                "id": "ask",
                "question": "Qual é a cor favorita atual da usuária?",
                "reference_answer": "A cor favorita atual é ocre.",
                "relevant_fact_ids": ["cor-nova"],
                "type": "fact_update",
            }
        ],
    }


def _distractor() -> dict[str, Any]:
    return {
        "id": "dlg-distrator",
        "turns": [
            {"role": "user", "content": "A cor da mochila da usuária é azul-petróleo."},
            {"role": "assistant", "content": "Anotado."},
            {"role": "user", "content": "A cor da caneca na mesa é verde-limão."},
            {"role": "assistant", "content": "Anotado o outro objeto."},
            {"role": "user", "content": "Qual é a cor da mochila da usuária?"},
        ],
        "planted_facts": [
            {"id": "mochila", "text": "azul-petróleo", "turn": 0, "kind": "target"},
            {"id": "caneca", "text": "verde-limão", "turn": 2, "kind": "distractor"},
        ],
        "probes": [
            {
                "id": "ask",
                "question": "Qual é a cor da mochila da usuária?",
                "reference_answer": "A cor da mochila é azul-petróleo.",
                "relevant_fact_ids": ["mochila"],
                "type": "distractor",
            }
        ],
    }


def _write(tmp_path: Path, *records: dict[str, Any]) -> Path:
    path = tmp_path / "dialogues.jsonl"
    _write_at(path, *records)
    return path


def _write_at(path: Path, *records: dict[str, Any]) -> None:
    payload = "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records)
    path.write_text(payload, encoding="utf-8")
