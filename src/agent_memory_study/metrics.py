"""Correção, recall e rótulos de falha de uma sondagem, e a tabela agregada.

O gabarito casa por igualdade depois da normalização. Paráfrase só entra se
um juiz for ligado. Recall é literal: o texto do fato plantado está no
contexto que a estratégia devolveu, sem a pergunta da sondagem.
"""

from __future__ import annotations

import csv
import io
import json
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from agent_memory_study.eval_dialogues import PROBE_TYPES, EvalDialogue, EvalProbe
from agent_memory_study.memory.summary import SUMMARY_PREFIX
from agent_memory_study.types import Message
from agent_memory_study.usage import Completion

FAILURE_MODES = ("forgotten_fact", "stale_summary", "retrieved_distractor")
REPORT_DISCLAIMER = "Traço desta execução. Não é resultado publicado do estudo."
REPORT_FAILURE_NOTE = (
    "Cada rótulo conta sondagens em que ele apareceu. Uma sondagem pode somar em mais de um rótulo."
)

_CSV_COLUMNS = (
    "estrategia",
    "tipo",
    "n",
    "acuracia",
    "recall_fato",
    "fato_esquecido",
    "resumo_obsoleto",
    "distrator_recuperado",
    "tokens_prompt",
    "tokens_resposta",
    "latencia_media_ms",
    "custo_estimado_usd",
)
_MD_HEADERS = (
    "estratégia",
    "tipo",
    "n",
    "acurácia",
    "recall do fato",
    "fato esquecido",
    "resumo obsoleto",
    "distrator recuperado",
    "tokens de prompt",
    "tokens de resposta",
    "latência média (ms)",
    "custo estimado (USD)",
)
_YES = frozenset({"sim", "yes"})
_NO = frozenset({"nao", "não", "no"})


@dataclass(frozen=True)
class ProbeRecord:
    """Uma sondagem já jogada numa estratégia."""

    dialogue_id: str
    probe_id: str
    probe_type: str
    strategy: str
    answer: str
    reference_answer: str
    correct: bool
    graded_by: str
    fact_recall: float
    relevant_fact_ids: tuple[str, ...]
    available_fact_ids: tuple[str, ...]
    failure_modes: tuple[str, ...]
    prompt_tokens: int
    completion_tokens: int
    answer_prompt_tokens: int
    answer_completion_tokens: int
    memory_prompt_tokens: int
    memory_completion_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    cached: bool
    model: str


@dataclass(frozen=True)
class AggregateRow:
    """Médias e somas de um par estratégia × tipo, ou tipo `all`."""

    strategy: str
    probe_type: str
    n: int
    accuracy: float
    fact_recall: float
    forgotten_fact: int
    stale_summary: int
    retrieved_distractor: int
    prompt_tokens: int
    completion_tokens: int
    mean_latency_ms: float
    estimated_cost_usd: float


class _Chat(Protocol):
    def complete(self, messages: Sequence[Message]) -> Completion:
        """Gera uma fala. O juiz só lê `text`."""


class Judge(Protocol):
    def equivalent(self, *, question: str, reference: str, answer: str) -> bool:
        """True quando a paráfrase preserva o literal do gabarito."""


class LLMJudge:
    """Juiz opcional. SIM ou NAO; qualquer outra fala conta como não equivalente."""

    def __init__(self, client: _Chat) -> None:
        self._client = client

    def equivalent(self, *, question: str, reference: str, answer: str) -> bool:
        completion = self._client.complete(
            [
                Message(
                    "system",
                    "Julgue se a resposta preserva os mesmos valores factuais do gabarito. "
                    "Responda apenas SIM ou NAO. "
                    "SIM somente se cada literal do gabarito estiver afirmado. "
                    "NAO se faltar um literal, se o valor for outro, "
                    "ou se o valor antigo substituído reaparecer.",
                ),
                Message(
                    "user",
                    f"Pergunta:\n{question}\n\nGabarito:\n{reference}\n\nResposta:\n{answer}",
                ),
            ]
        )
        return parse_judge_verdict(completion.text) is True


def normalize_answer(text: str) -> str:
    """NFKC, casefold, pontuação vira espaço. Hífen também, para o literal casar."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    cleaned = [
        char if char.isalnum() or unicodedata.category(char).startswith("M") else " "
        for char in normalized
    ]
    return " ".join("".join(cleaned).split())


def normalized_match(answer: str, reference: str) -> bool:
    """Igualdade após normalizar. Os dois vazios não contam como acerto."""

    left = normalize_answer(answer)
    right = normalize_answer(reference)
    return bool(right) and left == right


def parse_judge_verdict(text: str) -> bool | None:
    """Lê o primeiro token. None quando a fala não é SIM nem NAO."""

    tokens = normalize_answer(text).split()
    if not tokens:
        return None
    head = tokens[0]
    if head in _YES:
        return True
    if head in _NO:
        return False
    return None


def locate_probe_turn(dialogue: EvalDialogue, probe: EvalProbe) -> int:
    matches = [
        index
        for index, turn in enumerate(dialogue.turns)
        if turn.role == "user" and turn.content == probe.question
    ]
    if len(matches) != 1:
        raise ValueError(f"{dialogue.id}: probe {probe.id} is not exactly one user turn")
    return matches[0]


def score_probe(
    dialogue: EvalDialogue,
    probe: EvalProbe,
    context: Sequence[Message],
    *,
    strategy: str,
    answer_completion: Completion,
    latency_ms: float,
    memory_prompt_tokens: int = 0,
    memory_completion_tokens: int = 0,
    memory_cost_usd: float = 0.0,
    judge: Judge | None = None,
) -> ProbeRecord:
    """Mede uma sondagem contra o contexto já materializado, sem chamar memória."""

    available = _available_fact_ids(dialogue, probe, context)
    modes = _failure_modes(dialogue, probe, context, available)
    if not probe.relevant_fact_ids:
        raise ValueError(f"{dialogue.id}: probe {probe.id} has no relevant facts")
    answer = answer_completion.text
    if normalized_match(answer, probe.reference_answer):
        correct = True
        graded_by = "normalized"
    elif judge is not None and judge.equivalent(
        question=probe.question,
        reference=probe.reference_answer,
        answer=answer,
    ):
        correct = True
        graded_by = "judge"
    else:
        correct = False
        graded_by = "none"
    recall = len(available) / len(probe.relevant_fact_ids)
    return ProbeRecord(
        dialogue_id=dialogue.id,
        probe_id=probe.id,
        probe_type=probe.type,
        strategy=strategy,
        answer=answer,
        reference_answer=probe.reference_answer,
        correct=correct,
        graded_by=graded_by,
        fact_recall=recall,
        relevant_fact_ids=probe.relevant_fact_ids,
        available_fact_ids=available,
        failure_modes=modes,
        prompt_tokens=answer_completion.prompt_tokens + memory_prompt_tokens,
        completion_tokens=answer_completion.completion_tokens + memory_completion_tokens,
        answer_prompt_tokens=answer_completion.prompt_tokens,
        answer_completion_tokens=answer_completion.completion_tokens,
        memory_prompt_tokens=memory_prompt_tokens,
        memory_completion_tokens=memory_completion_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=answer_completion.estimated_cost_usd + memory_cost_usd,
        cached=answer_completion.cached,
        model=answer_completion.model,
    )


def aggregate(records: Sequence[ProbeRecord]) -> list[AggregateRow]:
    """Uma linha por tipo presente e uma linha `all` por estratégia."""

    strategies: list[str] = []
    for record in records:
        if record.strategy not in strategies:
            strategies.append(record.strategy)
    rows: list[AggregateRow] = []
    for strategy in strategies:
        scoped = [record for record in records if record.strategy == strategy]
        probe_types = [probe_type for probe_type in PROBE_TYPES if _has_type(scoped, probe_type)]
        extras: list[str] = []
        for record in scoped:
            if record.probe_type not in PROBE_TYPES and record.probe_type not in extras:
                extras.append(record.probe_type)
        for probe_type in [*probe_types, *extras]:
            rows.append(_aggregate_group(strategy, probe_type, _of_type(scoped, probe_type)))
        rows.append(_aggregate_group(strategy, "all", scoped))
    return rows


def render_csv(rows: Sequence[AggregateRow]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(_CSV_COLUMNS)
    for row in rows:
        writer.writerow(_row_cells(row))
    return buffer.getvalue()


def render_markdown(
    rows: Sequence[AggregateRow],
    *,
    provider: str,
    model: str,
    dialogue_count: int,
    strategies: Sequence[str],
    judge: bool,
) -> str:
    strategy_list = ", ".join(strategies) if strategies else "(nenhuma)"
    judge_label = "ligado" if judge else "desligado"
    lines = [
        "# Execução do harness",
        "",
        REPORT_DISCLAIMER,
        "",
        REPORT_FAILURE_NOTE,
        "",
        f"- provedor: `{provider}`",
        f"- modelo: `{model}`",
        f"- diálogos: {dialogue_count}",
        f"- estratégias: {strategy_list}",
        f"- juiz: {judge_label}",
        "",
        "| " + " | ".join(_MD_HEADERS) + " |",
        "| " + " | ".join("---" for _header in _MD_HEADERS) + " |",
    ]
    for row in rows:
        cells = [_md_cell(value) for value in _row_cells(row)]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def write_reports(
    directory: Path,
    records: Sequence[ProbeRecord],
    rows: Sequence[AggregateRow],
    *,
    provider: str,
    model: str,
    dialogue_count: int,
    strategies: Sequence[str],
    judge: bool,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(asdict(record), ensure_ascii=False) for record in records]
    body = "\n".join(lines)
    if body:
        body += "\n"
    (directory / "records.jsonl").write_text(body, encoding="utf-8")
    (directory / "summary.csv").write_text(render_csv(rows), encoding="utf-8")
    (directory / "summary.md").write_text(
        render_markdown(
            rows,
            provider=provider,
            model=model,
            dialogue_count=dialogue_count,
            strategies=strategies,
            judge=judge,
        ),
        encoding="utf-8",
    )


def _available_fact_ids(
    dialogue: EvalDialogue,
    probe: EvalProbe,
    context: Sequence[Message],
) -> tuple[str, ...]:
    by_id = {fact.id: fact for fact in dialogue.planted_facts}
    blob = _context_blob(context)
    available: list[str] = []
    for fact_id in probe.relevant_fact_ids:
        fact = by_id.get(fact_id)
        if fact is None:
            raise ValueError(f"{dialogue.id}: probe {probe.id} cites unknown fact {fact_id!r}")
        if fact.text in blob:
            available.append(fact_id)
    return tuple(available)


def _failure_modes(
    dialogue: EvalDialogue,
    probe: EvalProbe,
    context: Sequence[Message],
    available: tuple[str, ...],
) -> tuple[str, ...]:
    modes: list[str] = []
    if len(available) != len(probe.relevant_fact_ids):
        modes.append("forgotten_fact")
    if _summary_is_stale(dialogue, probe, context):
        modes.append("stale_summary")
    if _distractor_was_retrieved(dialogue, probe, context):
        modes.append("retrieved_distractor")
    return tuple(modes)


def _summary_is_stale(
    dialogue: EvalDialogue,
    probe: EvalProbe,
    context: Sequence[Message],
) -> bool:
    # Janela literal com o valor antigo não é resumo obsoleto: a correção ainda
    # pode estar ao lado. O rótulo só olha o texto já condensado.
    if probe.type != "fact_update" or len(probe.relevant_fact_ids) != 1:
        return False
    by_id = {fact.id: fact for fact in dialogue.planted_facts}
    current = by_id.get(probe.relevant_fact_ids[0])
    if current is None or current.supersedes is None:
        return False
    replaced = by_id.get(current.supersedes)
    if replaced is None:
        raise ValueError(
            f"{dialogue.id}: fact {current.id!r} supersedes unknown fact {current.supersedes!r}"
        )
    return replaced.text in _summary_blob(context)


def _distractor_was_retrieved(
    dialogue: EvalDialogue,
    probe: EvalProbe,
    context: Sequence[Message],
) -> bool:
    if probe.type != "distractor":
        return False
    by_id = {fact.id: fact for fact in dialogue.planted_facts}
    targets = [by_id[fact_id] for fact_id in probe.relevant_fact_ids if fact_id in by_id]
    if not targets:
        return False
    target_turn = min(fact.turn for fact in targets)
    probe_turn = locate_probe_turn(dialogue, probe)
    blob = _context_blob(context)
    return any(
        fact.kind == "distractor" and target_turn < fact.turn < probe_turn and fact.text in blob
        for fact in dialogue.planted_facts
    )


def _context_blob(context: Sequence[Message]) -> str:
    return "\n".join(message.content for message in context)


def _summary_blob(context: Sequence[Message]) -> str:
    chunks = [
        message.content[len(SUMMARY_PREFIX) :]
        for message in context
        if message.role == "system" and message.content.startswith(SUMMARY_PREFIX)
    ]
    return "\n".join(chunks)


def _aggregate_group(
    strategy: str,
    probe_type: str,
    records: Sequence[ProbeRecord],
) -> AggregateRow:
    count = len(records)
    if count == 0:
        raise ValueError("cannot aggregate an empty group")
    return AggregateRow(
        strategy=strategy,
        probe_type=probe_type,
        n=count,
        accuracy=sum(1.0 if record.correct else 0.0 for record in records) / count,
        fact_recall=sum(record.fact_recall for record in records) / count,
        forgotten_fact=sum("forgotten_fact" in record.failure_modes for record in records),
        stale_summary=sum("stale_summary" in record.failure_modes for record in records),
        retrieved_distractor=sum(
            "retrieved_distractor" in record.failure_modes for record in records
        ),
        prompt_tokens=sum(record.prompt_tokens for record in records),
        completion_tokens=sum(record.completion_tokens for record in records),
        mean_latency_ms=sum(record.latency_ms for record in records) / count,
        estimated_cost_usd=sum(record.estimated_cost_usd for record in records),
    )


def _has_type(records: Sequence[ProbeRecord], probe_type: str) -> bool:
    return any(record.probe_type == probe_type for record in records)


def _of_type(records: Sequence[ProbeRecord], probe_type: str) -> list[ProbeRecord]:
    return [record for record in records if record.probe_type == probe_type]


def _row_cells(row: AggregateRow) -> tuple[object, ...]:
    return (
        row.strategy,
        row.probe_type,
        row.n,
        _number(row.accuracy),
        _number(row.fact_recall),
        row.forgotten_fact,
        row.stale_summary,
        row.retrieved_distractor,
        row.prompt_tokens,
        row.completion_tokens,
        _number(row.mean_latency_ms),
        _number(row.estimated_cost_usd),
    )


def _number(value: float) -> str:
    return format(value, ".8g")


def _md_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
