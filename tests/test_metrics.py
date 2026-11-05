"""Contrato das métricas: normalização, recall literal, rótulos e agregação.

Os números daqui são contas sobre registros construídos no teste. Não são
resultado do estudo.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from agent_memory_study.config import load_settings
from agent_memory_study.eval_dialogues import EvalDialogue, load_eval_dialogues
from agent_memory_study.memory.summary import SUMMARY_PREFIX
from agent_memory_study.metrics import (
    REPORT_DISCLAIMER,
    REPORT_FAILURE_NOTE,
    LLMJudge,
    ProbeRecord,
    aggregate,
    normalize_answer,
    normalized_match,
    parse_judge_verdict,
    score_probe,
    write_reports,
)
from agent_memory_study.types import Message
from agent_memory_study.usage import Completion, estimate_cost_usd


@pytest.fixture(scope="module")
def dialogues() -> dict[str, EvalDialogue]:
    return {dialogue.id: dialogue for dialogue in load_eval_dialogues()}


def test_normalized_match_ignores_case_punctuation_and_hyphen() -> None:
    assert normalized_match("A Cor Favorita é azul-petróleo.", "a cor favorita é azul petróleo")
    assert normalize_answer("  Azul-Petróleo! ") == "azul petróleo"
    assert not normalized_match("azul petróleo", "A cor favorita é azul-petróleo.")
    assert not normalized_match("!!!", "???")


def test_parse_judge_verdict_reads_the_first_token_only() -> None:
    assert parse_judge_verdict("SIM.") is True
    assert parse_judge_verdict("YES they match") is True
    assert parse_judge_verdict("NÃO, o valor mudou") is False
    assert parse_judge_verdict("nao") is False
    assert parse_judge_verdict("talvez") is None
    assert parse_judge_verdict("   ") is None


def test_llm_judge_asks_for_sim_or_nao_and_rejects_other_text() -> None:
    class _Client:
        def __init__(self, text: str) -> None:
            self.text = text
            self.prompts: list[list[Message]] = []

        def complete(self, messages: list[Message]) -> Completion:
            self.prompts.append(list(messages))
            return _completion(self.text)

    client = _Client("SIM")
    assert LLMJudge(client).equivalent(question="pergunta", reference="gabarito", answer="fala")
    sent = "\n".join(message.content for message in client.prompts[0])
    assert "gabarito" in sent
    assert "fala" in sent
    assert (
        LLMJudge(_Client("talvez")).equivalent(
            question="pergunta",
            reference="gabarito",
            answer="fala",
        )
        is False
    )


def test_exact_match_does_not_call_the_judge(dialogues: dict[str, EvalDialogue]) -> None:
    dialogue = dialogues["dlg-01-cor-imediata"]
    probe = dialogue.probes[0]
    judge = _CountingJudge(verdict=True)

    record = score_probe(
        dialogue,
        probe,
        [dialogue.turns[0]],
        strategy="window",
        answer_completion=_completion(probe.reference_answer.upper()),
        latency_ms=1,
        judge=judge,
    )

    assert record.correct is True
    assert record.graded_by == "normalized"
    assert judge.calls == 0


def test_judge_accepts_a_paraphrase_when_normalization_fails(
    dialogues: dict[str, EvalDialogue],
) -> None:
    dialogue = dialogues["dlg-01-cor-imediata"]
    probe = dialogue.probes[0]
    judge = _CountingJudge(verdict=True)

    record = score_probe(
        dialogue,
        probe,
        [dialogue.turns[0]],
        strategy="window",
        answer_completion=_completion("outra frase"),
        latency_ms=1,
        judge=judge,
    )

    assert record.correct is True
    assert record.graded_by == "judge"
    assert judge.calls == 1
    assert judge.last == (probe.question, probe.reference_answer, "outra frase")


def test_multi_hop_recall_is_the_fraction_of_literals_present(
    dialogues: dict[str, EvalDialogue],
) -> None:
    dialogue = dialogues["dlg-16-chave-bloco"]
    probe = dialogue.probes[0]
    bloco = next(fact for fact in dialogue.planted_facts if fact.id == "bloco")

    record = score_probe(
        dialogue,
        probe,
        [dialogue.turns[bloco.turn]],
        strategy="window",
        answer_completion=_completion("não sei"),
        latency_ms=4,
    )

    assert record.fact_recall == 0.5
    assert record.available_fact_ids == ("bloco",)
    assert record.failure_modes == ("forgotten_fact",)
    assert record.correct is False


def test_stale_summary_requires_the_replaced_literal_inside_the_summary(
    dialogues: dict[str, EvalDialogue],
) -> None:
    dialogue = dialogues["dlg-11-cor-corrigida"]
    probe = dialogue.probes[0]
    current = next(fact for fact in dialogue.planted_facts if fact.id == probe.relevant_fact_ids[0])
    replaced = next(fact for fact in dialogue.planted_facts if fact.id == current.supersedes)
    summarized = [
        Message("system", f"{SUMMARY_PREFIX}a cor era {replaced.text}"),
        Message("user", f"agora é {current.text}"),
    ]
    literal_only = [Message("user", f"a cor era {replaced.text}")]

    stale = score_probe(
        dialogue,
        probe,
        summarized,
        strategy="summary",
        answer_completion=_completion("x"),
        latency_ms=0,
    )
    window = score_probe(
        dialogue,
        probe,
        literal_only,
        strategy="window",
        answer_completion=_completion("x"),
        latency_ms=0,
    )

    assert stale.failure_modes == ("stale_summary",)
    assert stale.fact_recall == 1
    assert window.failure_modes == ("forgotten_fact",)


def test_retrieved_distractor_is_only_labeled_on_distractor_probes(
    dialogues: dict[str, EvalDialogue],
) -> None:
    dialogue = dialogues["dlg-21-mochila-caneca"]
    probe = dialogue.probes[0]
    target = next(fact for fact in dialogue.planted_facts if fact.id in probe.relevant_fact_ids)
    distractor = next(fact for fact in dialogue.planted_facts if fact.kind == "distractor")
    both = score_probe(
        dialogue,
        probe,
        [dialogue.turns[target.turn], dialogue.turns[distractor.turn]],
        strategy="vector",
        answer_completion=_completion("x"),
        latency_ms=0,
    )
    target_only = score_probe(
        dialogue,
        probe,
        [dialogue.turns[target.turn]],
        strategy="vector",
        answer_completion=_completion("x"),
        latency_ms=0,
    )
    immediate = dialogues["dlg-01-cor-imediata"]
    not_a_distractor_probe = score_probe(
        immediate,
        immediate.probes[0],
        [Message("user", f"{immediate.planted_facts[0].text} verde-limão")],
        strategy="window",
        answer_completion=_completion("x"),
        latency_ms=0,
    )

    assert both.failure_modes == ("retrieved_distractor",)
    assert both.fact_recall == 1
    assert target_only.failure_modes == ()
    assert "retrieved_distractor" not in not_a_distractor_probe.failure_modes


def test_aggregate_means_sums_and_counts_each_failure_label() -> None:
    records = [
        _record(probe_id="a", probe_type="immediate_recall", correct=True, fact_recall=1),
        _record(
            probe_id="b",
            probe_type="delayed_recall",
            correct=False,
            fact_recall=0.5,
            prompt_tokens=20,
            completion_tokens=6,
            latency_ms=30,
            estimated_cost_usd=0.2,
            failure_modes=("forgotten_fact", "stale_summary"),
        ),
        _record(
            probe_id="c",
            strategy="summary",
            correct=True,
            prompt_tokens=5,
            completion_tokens=1,
            latency_ms=0,
            estimated_cost_usd=0,
            failure_modes=("retrieved_distractor",),
        ),
    ]

    rows = {(_row.strategy, _row.probe_type): _row for _row in aggregate(records)}
    window_all = rows[("window", "all")]

    assert window_all.n == 2
    assert window_all.accuracy == pytest.approx(0.5)
    assert window_all.fact_recall == pytest.approx(0.75)
    assert window_all.forgotten_fact == 1
    assert window_all.stale_summary == 1
    assert window_all.retrieved_distractor == 0
    assert window_all.prompt_tokens == 30
    assert window_all.completion_tokens == 10
    assert window_all.mean_latency_ms == pytest.approx(20)
    assert window_all.estimated_cost_usd == pytest.approx(0.3)
    assert rows[("window", "immediate_recall")].n == 1
    assert rows[("window", "delayed_recall")].forgotten_fact == 1
    assert rows[("summary", "all")].retrieved_distractor == 1


def test_csv_and_markdown_round_trip_the_aggregate_without_claiming_a_study_result(
    tmp_path: Path,
) -> None:
    rows = aggregate(
        [
            _record(correct=True, fact_recall=1, estimated_cost_usd=0.1),
            _record(
                probe_id="b",
                probe_type="delayed_recall",
                correct=False,
                fact_recall=0,
                failure_modes=("forgotten_fact",),
                estimated_cost_usd=0.2,
                latency_ms=30,
                prompt_tokens=20,
            ),
        ]
    )
    write_reports(
        tmp_path,
        [_record()],
        rows,
        provider="fake",
        model="fake",
        dialogue_count=2,
        strategies=("window",),
        judge=False,
    )
    parsed = list(csv.DictReader((tmp_path / "summary.csv").open(encoding="utf-8")))
    markdown = (tmp_path / "summary.md").read_text(encoding="utf-8")
    window_all = next(row for row in parsed if row["tipo"] == "all")

    assert float(window_all["acuracia"]) == pytest.approx(0.5)
    assert float(window_all["recall_fato"]) == pytest.approx(0.5)
    assert int(window_all["fato_esquecido"]) == 1
    assert int(window_all["n"]) == 2
    assert REPORT_DISCLAIMER in markdown
    assert REPORT_FAILURE_NOTE in markdown
    assert "resultado publicado" in markdown
    assert "window" in markdown
    assert "delayed_recall" in markdown


def test_list_price_and_override_do_not_invent_a_price_for_an_unknown_model() -> None:
    listed = load_settings({})
    overridden = load_settings(
        {
            "OPENAI_MODEL": "gpt-4o-mini",
            "PROMPT_USD_PER_MILLION": "2",
            "COMPLETION_USD_PER_MILLION": "4",
        }
    )

    assert estimate_cost_usd(listed, "gpt-4o-mini", 1_000_000, 1_000_000, cached=False) == (
        pytest.approx(0.75)
    )
    assert estimate_cost_usd(listed, "fake", 500, 500, cached=False) == 0
    assert estimate_cost_usd(listed, "gpt-4o-mini", 10, 10, cached=True) == 0
    assert estimate_cost_usd(listed, "unknown-model", 1_000_000, 1_000_000, cached=False) == 0
    assert estimate_cost_usd(overridden, "gpt-4o-mini", 1_000_000, 1_000_000, cached=False) == (
        pytest.approx(6)
    )
    assert estimate_cost_usd(
        overridden,
        "text-embedding-3-small",
        1_000_000,
        0,
        cached=False,
    ) == pytest.approx(0.02)


class _CountingJudge:
    def __init__(self, verdict: bool) -> None:
        self.verdict = verdict
        self.calls = 0
        self.last: tuple[str, str, str] | None = None

    def equivalent(self, *, question: str, reference: str, answer: str) -> bool:
        self.calls += 1
        self.last = (question, reference, answer)
        return self.verdict


def _completion(text: str, *, prompt_tokens: int = 3, completion_tokens: int = 1) -> Completion:
    return Completion(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=0.0,
        cached=False,
        model="fake",
    )


def _record(**overrides: object) -> ProbeRecord:
    payload: dict[str, object] = {
        "dialogue_id": "d",
        "probe_id": "p",
        "probe_type": "immediate_recall",
        "strategy": "window",
        "answer": "a",
        "reference_answer": "a",
        "correct": True,
        "graded_by": "normalized",
        "fact_recall": 1.0,
        "relevant_fact_ids": ("f",),
        "available_fact_ids": ("f",),
        "failure_modes": (),
        "prompt_tokens": 10,
        "completion_tokens": 4,
        "answer_prompt_tokens": 10,
        "answer_completion_tokens": 4,
        "memory_prompt_tokens": 0,
        "memory_completion_tokens": 0,
        "latency_ms": 10.0,
        "estimated_cost_usd": 0.1,
        "cached": False,
        "model": "fake",
    }
    payload.update(overrides)
    return ProbeRecord(**payload)  # type: ignore[arg-type]
