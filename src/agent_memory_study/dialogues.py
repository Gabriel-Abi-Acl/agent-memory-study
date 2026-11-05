"""Carrega o fixture sintético e recusa diálogo que quebra o contrato.

Fato plantado tem de aparecer na mensagem indicada, e a sondagem só
pode citar fato já ocorrido até `after_turn`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from agent_memory_study.types import Dialogue, Message, PlantedFact, Probe

_ROLES = frozenset({"user", "assistant", "system"})


def default_dialogues_path() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "dialogues.json"


def load_dialogues(path: Path | None = None) -> tuple[Dialogue, ...]:
    source = path or default_dialogues_path()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not isinstance(payload.get("dialogues"), list):
        raise ValueError("dialogues file must be an object with a 'dialogues' list")
    raw_items = cast(list[Any], payload["dialogues"])
    if not raw_items:
        raise ValueError("dialogues file is empty")
    dialogues = tuple(_parse_dialogue(item, index) for index, item in enumerate(raw_items))
    ids = [dialogue.id for dialogue in dialogues]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate dialogue id")
    return dialogues


def _parse_dialogue(raw: Any, index: int) -> Dialogue:
    item = _as_dict(raw, f"dialogues[{index}]")
    dialogue_id = _text(item, "id", f"dialogues[{index}]")
    messages = _parse_messages(item.get("messages"), dialogue_id)
    facts = _parse_facts(item.get("facts"), dialogue_id, messages)
    probes = _parse_probes(item.get("probes"), dialogue_id, messages, facts)
    return Dialogue(id=dialogue_id, messages=messages, facts=facts, probes=probes)


def _parse_messages(raw: Any, dialogue_id: str) -> tuple[Message, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{dialogue_id}: messages must be a non-empty list")
    messages: list[Message] = []
    for index, candidate in enumerate(raw):
        item = _as_dict(candidate, f"{dialogue_id}.messages[{index}]")
        role = item.get("role")
        content = item.get("content")
        if role not in _ROLES or not isinstance(content, str) or not content.strip():
            raise ValueError(f"{dialogue_id}.messages[{index}] has an invalid role or content")
        messages.append(Message(role=role, content=content))
    return tuple(messages)


def _parse_facts(
    raw: Any,
    dialogue_id: str,
    messages: tuple[Message, ...],
) -> tuple[PlantedFact, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{dialogue_id}: facts must be a non-empty list")
    facts: list[PlantedFact] = []
    for index, candidate in enumerate(raw):
        item = _as_dict(candidate, f"{dialogue_id}.facts[{index}]")
        fact_id = _text(item, "id", f"{dialogue_id}.facts[{index}]")
        text = _text(item, "text", f"{dialogue_id}.facts[{index}]")
        turn = _index(item.get("planted_at_turn"), len(messages), f"{dialogue_id}.facts[{index}]")
        if text not in messages[turn].content:
            raise ValueError(f"{dialogue_id}: fact {fact_id!r} is not in message {turn}")
        facts.append(PlantedFact(id=fact_id, text=text, planted_at_turn=turn))
    if len({fact.id for fact in facts}) != len(facts):
        raise ValueError(f"{dialogue_id}: duplicate fact id")
    return tuple(facts)


def _parse_probes(
    raw: Any,
    dialogue_id: str,
    messages: tuple[Message, ...],
    facts: tuple[PlantedFact, ...],
) -> tuple[Probe, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{dialogue_id}: probes must be a non-empty list")
    by_id = {fact.id: fact for fact in facts}
    probes: list[Probe] = []
    for index, candidate in enumerate(raw):
        item = _as_dict(candidate, f"{dialogue_id}.probes[{index}]")
        probe_id = _text(item, "id", f"{dialogue_id}.probes[{index}]")
        question = _text(item, "question", f"{dialogue_id}.probes[{index}]")
        after_turn = _index(
            item.get("after_turn"),
            len(messages),
            f"{dialogue_id}.probes[{index}]",
        )
        expected = item.get("expected_fact_ids")
        if (
            not isinstance(expected, list)
            or not expected
            or not all(isinstance(fact_id, str) and fact_id for fact_id in expected)
        ):
            raise ValueError(f"{dialogue_id}.probes[{index}] has invalid expected_fact_ids")
        for fact_id in expected:
            fact = by_id.get(fact_id)
            if fact is None:
                raise ValueError(
                    f"{dialogue_id}: probe {probe_id!r} cites unknown fact {fact_id!r}"
                )
            if fact.planted_at_turn > after_turn:
                raise ValueError(
                    f"{dialogue_id}: probe {probe_id!r} is asked before fact {fact_id!r}"
                )
        probes.append(
            Probe(
                id=probe_id,
                question=question,
                after_turn=after_turn,
                expected_fact_ids=tuple(expected),
            )
        )
    if len({probe.id for probe in probes}) != len(probes):
        raise ValueError(f"{dialogue_id}: duplicate probe id")
    return tuple(probes)


def _as_dict(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _text(item: dict[str, Any], key: str, label: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _index(value: Any, size: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < size:
        raise ValueError(f"{label} has an index outside the dialogue")
    return value
