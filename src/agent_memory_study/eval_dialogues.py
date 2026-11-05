"""Dataset versionado de avaliação e a validação offline do esquema.

Cada linha de `data/eval/dialogues.v1.jsonl` é um diálogo com `turns`,
`planted_facts` e `probes` (pergunta, gabarito, fatos citados e tipo).
O carregador só lê arquivo local: sem rede e sem modelo.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from agent_memory_study.types import Message, Role

# Recall imediato: a pergunta ainda está colada no fato.
# Recall tardio: abaixo desta lacuna a sondagem não exercita expulsão da janela.
IMMEDIATE_MAX_GAP = 2
DELAYED_MIN_GAP = 8

ProbeType = Literal[
    "immediate_recall",
    "delayed_recall",
    "fact_update",
    "multi_hop",
    "distractor",
]
FactKind = Literal["target", "distractor"]

PROBE_TYPES: tuple[ProbeType, ...] = (
    "immediate_recall",
    "delayed_recall",
    "fact_update",
    "multi_hop",
    "distractor",
)
_ROLES = frozenset({"user", "assistant", "system"})
_KINDS = frozenset({"target", "distractor"})


@dataclass(frozen=True)
class EvalFact:
    """Literal que vive num único turno do diálogo.

    `distractor` é um valor concorrente: o gabarito não pode adotá-lo.
    `supersedes` é o alvo anterior que este valor substitui.
    """

    id: str
    text: str
    turn: int
    kind: FactKind
    supersedes: str | None = None


@dataclass(frozen=True)
class EvalProbe:
    """Pergunta com gabarito, ancorada num turno `user` já existente.

    `relevant_fact_ids` só pode citar fatos `target` anteriores a essa pergunta.
    """

    id: str
    question: str
    reference_answer: str
    relevant_fact_ids: tuple[str, ...]
    type: ProbeType


@dataclass(frozen=True)
class EvalDialogue:
    """Conversa fechada: fatos e sondagens já estão nos turnos.

    A pergunta da sondagem é um turno `user`, não um apêndice fora do diálogo.
    """

    id: str
    turns: tuple[Message, ...]
    planted_facts: tuple[EvalFact, ...]
    probes: tuple[EvalProbe, ...]


# O estudo lê esta revisão. Outra versão do arquivo precisa de outro carregador.
def default_eval_dialogues_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "eval" / "dialogues.v1.jsonl"


# Lê o JSONL local e recusa diálogo que quebra o contrato do v1.
# Sem essa checagem, uma sondagem anterior ao fato vira métrica furada.
def load_eval_dialogues(path: Path | None = None) -> tuple[EvalDialogue, ...]:
    source = path or default_eval_dialogues_path()
    lines = source.read_text(encoding="utf-8").splitlines()
    if not any(line.strip() for line in lines):
        raise ValueError("eval dialogues file is empty")
    dialogues: list[EvalDialogue] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            raise ValueError(f"line {line_number} is blank")
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"line {line_number} is not valid JSON") from exc
        dialogues.append(_parse_dialogue(payload, line_number))
    ids = [dialogue.id for dialogue in dialogues]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dialogue id")
    return tuple(dialogues)


def _parse_dialogue(raw: Any, line_number: int) -> EvalDialogue:
    label = f"line {line_number}"
    item = _as_dict(raw, label)
    dialogue_id = _text(item, "id", label)
    turns = _parse_turns(item.get("turns"), dialogue_id)
    facts = _parse_facts(item.get("planted_facts"), dialogue_id, turns)
    probes = _parse_probes(item.get("probes"), dialogue_id, turns, facts)
    return EvalDialogue(id=dialogue_id, turns=turns, planted_facts=facts, probes=probes)


def _parse_turns(raw: Any, dialogue_id: str) -> tuple[Message, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{dialogue_id}: turns must be a non-empty list")
    turns: list[Message] = []
    for index, candidate in enumerate(raw):
        item = _as_dict(candidate, f"{dialogue_id}.turns[{index}]")
        role = item.get("role")
        content = item.get("content")
        if role not in _ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{dialogue_id}.turns[{index}] has an invalid role or content")
        turns.append(Message(role=cast(Role, role), content=content))
    return tuple(turns)


def _parse_facts(
    raw: Any,
    dialogue_id: str,
    turns: tuple[Message, ...],
) -> tuple[EvalFact, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{dialogue_id}: planted_facts must be a non-empty list")
    facts: list[EvalFact] = []
    for index, candidate in enumerate(raw):
        label = f"{dialogue_id}.planted_facts[{index}]"
        item = _as_dict(candidate, label)
        fact_id = _text(item, "id", label)
        text = _text(item, "text", label)
        turn = _index(item.get("turn"), len(turns), label)
        kind = item.get("kind")
        if kind not in _KINDS:
            raise ValueError(f"{label}.kind must be 'target' or 'distractor'")
        if text not in turns[turn].content:
            raise ValueError(f"{dialogue_id}: fact {fact_id!r} is not in turn {turn}")
        for other, message in enumerate(turns):
            if other != turn and text in message.content:
                raise ValueError(f"{dialogue_id}: fact {fact_id!r} also appears in turn {other}")
        facts.append(
            EvalFact(
                id=fact_id,
                text=text,
                turn=turn,
                kind=cast(FactKind, kind),
                supersedes=_optional_id(item, "supersedes", label),
            )
        )
    if len({fact.id for fact in facts}) != len(facts):
        raise ValueError(f"{dialogue_id}: duplicate fact id")
    _check_supersedes(dialogue_id, tuple(facts))
    return tuple(facts)


def _parse_probes(
    raw: Any,
    dialogue_id: str,
    turns: tuple[Message, ...],
    facts: tuple[EvalFact, ...],
) -> tuple[EvalProbe, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{dialogue_id}: probes must be a non-empty list")
    by_id = {fact.id: fact for fact in facts}
    located: list[tuple[EvalProbe, int]] = []
    for index, candidate in enumerate(raw):
        label = f"{dialogue_id}.probes[{index}]"
        item = _as_dict(candidate, label)
        probe_id = _text(item, "id", label)
        question = _text(item, "question", label)
        reference = _text(item, "reference_answer", label)
        probe_type = item.get("type")
        if probe_type not in PROBE_TYPES:
            raise ValueError(f"{label} has an invalid type")
        relevant = _fact_ids(item.get("relevant_fact_ids"), label)
        for fact_id in relevant:
            if fact_id not in by_id:
                raise ValueError(
                    f"{dialogue_id}: probe {probe_id!r} cites unknown fact {fact_id!r}"
                )
        probe_turn = _user_turn(turns, question, label)
        probe = EvalProbe(
            id=probe_id,
            question=question,
            reference_answer=reference,
            relevant_fact_ids=tuple(relevant),
            type=cast(ProbeType, probe_type),
        )
        _check_probe_type(dialogue_id, probe, facts, by_id, probe_turn)
        located.append((probe, probe_turn))
    probes = tuple(probe for probe, _turn in located)
    if len({probe.id for probe in probes}) != len(probes):
        raise ValueError(f"{dialogue_id}: duplicate probe id")
    questions = [probe.question for probe in probes]
    if len(questions) != len(set(questions)):
        raise ValueError(f"{dialogue_id}: duplicate probe question")
    _check_target_facts_are_used(dialogue_id, facts, probes)
    _check_distractor_facts_are_in_play(dialogue_id, facts, located, by_id)
    return probes


def _check_supersedes(dialogue_id: str, facts: tuple[EvalFact, ...]) -> None:
    by_id = {fact.id: fact for fact in facts}
    for fact in facts:
        if fact.supersedes is None:
            continue
        if fact.kind != "target":
            raise ValueError(
                f"{dialogue_id}: fact {fact.id!r} supersedes another but is not a target"
            )
        seen = {fact.id}
        current = fact
        while current.supersedes is not None:
            parent = by_id.get(current.supersedes)
            if parent is None:
                raise ValueError(
                    f"{dialogue_id}: fact {current.id!r} supersedes unknown fact "
                    f"{current.supersedes!r}"
                )
            if parent.id in seen:
                raise ValueError(f"{dialogue_id}: fact {fact.id!r} has a supersedes cycle")
            if parent.kind != "target":
                raise ValueError(
                    f"{dialogue_id}: fact {current.id!r} supersedes non-target {parent.id!r}"
                )
            if parent.turn >= current.turn:
                raise ValueError(f"{dialogue_id}: fact {current.id!r} must replace an earlier fact")
            seen.add(parent.id)
            current = parent


# Geometria por tipo. Um rótulo sozinho não basta: `delayed_recall` colado
# no fato, ou `fact_update` sem valor novo, mediria outra coisa.
def _check_probe_type(
    dialogue_id: str,
    probe: EvalProbe,
    facts: tuple[EvalFact, ...],
    by_id: dict[str, EvalFact],
    probe_turn: int,
) -> None:
    cited = [by_id[fact_id] for fact_id in probe.relevant_fact_ids]
    for fact in cited:
        if fact.kind != "target":
            raise ValueError(f"{dialogue_id}: probe {probe.id!r} cites distractor fact {fact.id!r}")
        if fact.turn >= probe_turn:
            raise ValueError(
                f"{dialogue_id}: probe {probe.id!r} does not come after fact {fact.id!r}"
            )
        if fact.text not in probe.reference_answer:
            raise ValueError(f"{dialogue_id}: probe {probe.id!r} reference misses fact {fact.id!r}")

    if probe.type == "multi_hop":
        if len(cited) != 2:
            raise ValueError(f"{dialogue_id}: multi_hop probe must cite two facts")
        if cited[0].turn == cited[1].turn:
            raise ValueError(f"{dialogue_id}: multi_hop facts must come from two different turns")
        if any(fact.supersedes is not None for fact in cited):
            raise ValueError(f"{dialogue_id}: multi_hop probe must not cite a corrected fact")
        return

    if len(cited) != 1:
        raise ValueError(f"{dialogue_id}: {probe.type} probe must cite one fact")
    fact = cited[0]
    gap = probe_turn - fact.turn
    if probe.type == "immediate_recall":
        if gap > IMMEDIATE_MAX_GAP:
            raise ValueError(f"{dialogue_id}: immediate_recall probe is too far from its fact")
        _reject_corrected_fact(dialogue_id, probe, fact)
        return
    if probe.type == "delayed_recall":
        if gap < DELAYED_MIN_GAP:
            raise ValueError(f"{dialogue_id}: delayed_recall probe is too close to its fact")
        _reject_corrected_fact(dialogue_id, probe, fact)
        return
    if probe.type == "fact_update":
        if fact.supersedes is None:
            raise ValueError(
                f"{dialogue_id}: fact_update probe must cite a fact that supersedes another"
            )
        replaced = by_id[fact.supersedes]
        if replaced.text in probe.reference_answer:
            raise ValueError(
                f"{dialogue_id}: fact_update reference still contains the replaced fact"
            )
        return
    if probe.type == "distractor":
        _reject_corrected_fact(dialogue_id, probe, fact)
        between = [
            item
            for item in facts
            if item.kind == "distractor" and fact.turn < item.turn < probe_turn
        ]
        if not between:
            raise ValueError(
                f"{dialogue_id}: distractor probe has no distractor fact "
                "between the target and the question"
            )
        for item in between:
            if item.text in probe.reference_answer:
                raise ValueError(
                    f"{dialogue_id}: distractor reference contains distractor fact {item.id!r}"
                )
        return
    raise ValueError(f"{dialogue_id}: probe {probe.id!r} has an invalid type")


def _reject_corrected_fact(dialogue_id: str, probe: EvalProbe, fact: EvalFact) -> None:
    if fact.supersedes is not None:
        raise ValueError(
            f"{dialogue_id}: probe {probe.id!r} cites a corrected fact but is not fact_update"
        )


def _check_target_facts_are_used(
    dialogue_id: str,
    facts: tuple[EvalFact, ...],
    probes: tuple[EvalProbe, ...],
) -> None:
    cited = {fact_id for probe in probes for fact_id in probe.relevant_fact_ids}
    superseded = {fact.supersedes for fact in facts if fact.supersedes is not None}
    for fact in facts:
        if fact.kind != "target":
            continue
        if fact.id not in cited and fact.id not in superseded:
            raise ValueError(
                f"{dialogue_id}: target fact {fact.id!r} is neither cited nor superseded"
            )


def _check_distractor_facts_are_in_play(
    dialogue_id: str,
    facts: tuple[EvalFact, ...],
    located: list[tuple[EvalProbe, int]],
    by_id: dict[str, EvalFact],
) -> None:
    for fact in facts:
        if fact.kind != "distractor":
            continue
        covered = any(
            probe.type == "distractor" and by_id[fact_id].turn < fact.turn < probe_turn
            for probe, probe_turn in located
            for fact_id in probe.relevant_fact_ids
        )
        if not covered:
            raise ValueError(
                f"{dialogue_id}: distractor fact {fact.id!r} is not between a target and a probe"
            )


def _user_turn(turns: tuple[Message, ...], question: str, label: str) -> int:
    matches = [
        index
        for index, turn in enumerate(turns)
        if turn.role == "user" and turn.content == question
    ]
    if len(matches) != 1:
        raise ValueError(f"{label} question must match exactly one user turn")
    return matches[0]


def _fact_ids(raw: Any, label: str) -> list[str]:
    if (
        not isinstance(raw, list)
        or not raw
        or not all(isinstance(fact_id, str) and fact_id.strip() for fact_id in raw)
    ):
        raise ValueError(f"{label}.relevant_fact_ids must be a non-empty list of strings")
    fact_ids = cast(list[str], raw)
    if len(set(fact_ids)) != len(fact_ids):
        raise ValueError(f"{label}.relevant_fact_ids has a duplicate id")
    return fact_ids


def _as_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _text(item: dict[str, Any], key: str, label: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _optional_id(item: dict[str, Any], key: str, label: str) -> str | None:
    if key not in item or item[key] is None:
        return None
    value = item[key]
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string when present")
    return value


def _index(value: Any, size: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{label} has an index outside the dialogue")
    return value
