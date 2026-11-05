"""Contrato do runner: história scriptada, modelo só na sondagem, inspeção da memória.

A suíte não publica acurácia do estudo. O LLM fake não responde o gabarito,
salvo quando o próprio teste entrega o texto de referência.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest

from agent_memory_study.cli import main
from agent_memory_study.config import load_settings
from agent_memory_study.eval_dialogues import EvalDialogue, load_eval_dialogues
from agent_memory_study.llm import FakeLLM
from agent_memory_study.llm_runtime import (
    EmbedAccounting,
    EstimatingChatClient,
    RetryingChatClient,
    TransientLLMError,
)
from agent_memory_study.metrics import FAILURE_MODES, aggregate
from agent_memory_study.runner import run_evaluation
from agent_memory_study.types import Message
from agent_memory_study.usage import Completion, estimate_message_tokens


@pytest.fixture(scope="module")
def dialogues() -> dict[str, EvalDialogue]:
    return {dialogue.id: dialogue for dialogue in load_eval_dialogues()}


def test_runner_plays_a_few_dialogues_and_calls_the_model_only_at_probes(
    dialogues: dict[str, EvalDialogue],
) -> None:
    chosen = [dialogues["dlg-01-cor-imediata"], dialogues["dlg-16-chave-bloco"]]
    settings = load_settings({})
    fake = FakeLLM("resposta-fake")
    client = EstimatingChatClient(fake, model="fake", settings=settings)

    records = run_evaluation(chosen, ["window", "summary", "vector"], settings, client=client)

    assert len(records) == 6
    assert len(fake.prompts) == 6
    assert {record.strategy for record in records} == {"window", "summary", "vector"}
    immediate = [record for record in records if record.dialogue_id == "dlg-01-cor-imediata"]
    assert all(record.fact_recall == 1 for record in immediate)
    assert all(record.failure_modes == () for record in immediate)
    assert all(record.answer == "resposta-fake" for record in records)
    assert all(record.correct is False for record in records)
    assert sum(row.n for row in aggregate(records) if row.probe_type == "all") == 6


def test_memory_state_at_probe_time_sees_only_what_the_strategy_kept(
    dialogues: dict[str, EvalDialogue],
) -> None:
    settings = load_settings({})
    clock = _Clock()
    immediate = dialogues["dlg-01-cor-imediata"]
    forgotten = _run(
        immediate,
        "window",
        replace(settings, window_size=1),
        reply="não sei",
        clock=clock,
    )
    kept = _run(
        immediate,
        "window",
        replace(settings, window_size=8),
        reply="a cor favorita é azul-petróleo",
        clock=clock,
    )
    partial = _run(
        dialogues["dlg-16-chave-bloco"],
        "window",
        replace(settings, window_size=2),
        reply="não sei",
    )
    stale = _run(
        dialogues["dlg-11-cor-corrigida"],
        "summary",
        replace(settings, summary_recent_limit=1),
        reply="não sei",
    )
    exposed = _run(
        dialogues["dlg-21-mochila-caneca"],
        "window",
        replace(settings, window_size=8),
        reply="não sei",
    )
    evicted = _run(
        dialogues["dlg-21-mochila-caneca"],
        "window",
        replace(settings, window_size=1),
        reply="não sei",
    )

    fact = immediate.planted_facts[0].text
    assert forgotten.record.failure_modes == ("forgotten_fact",)
    assert forgotten.record.fact_recall == 0
    assert forgotten.record.latency_ms == pytest.approx(50)
    assert all(fact not in message.content for message in forgotten.prompts[0])
    assert forgotten.prompts[0][-1].content == immediate.probes[0].question
    assert len(forgotten.prompts) == 1
    assert kept.record.correct is True
    assert kept.record.graded_by == "normalized"
    assert kept.record.failure_modes == ()
    assert kept.record.available_fact_ids == immediate.probes[0].relevant_fact_ids
    assert any(
        fact in message.content and message.content != immediate.probes[0].question
        for message in kept.prompts[0]
    )
    assert partial.record.fact_recall == 0.5
    assert partial.record.available_fact_ids == ("bloco",)
    assert partial.record.failure_modes == ("forgotten_fact",)
    assert stale.record.failure_modes == ("stale_summary",)
    assert stale.record.fact_recall == 1
    assert exposed.record.failure_modes == ("retrieved_distractor",)
    assert exposed.record.fact_recall == 1
    assert evicted.record.failure_modes == ("forgotten_fact",)


def test_summary_calls_are_counted_apart_from_the_probe_answer(
    dialogues: dict[str, EvalDialogue],
) -> None:
    dialogue = dialogues["dlg-11-cor-corrigida"]
    settings = replace(load_settings({}), summary_recent_limit=1)
    fake = FakeLLM("resumo")
    client = EstimatingChatClient(fake, model="fake", settings=settings)

    records = run_evaluation(
        [dialogue],
        ["summary"],
        settings,
        client=client,
        summarize_with_client=True,
    )

    assert len(records) == 1
    assert len(fake.prompts) == 5
    record = records[0]
    assert record.answer_prompt_tokens == estimate_message_tokens(fake.prompts[3])
    assert record.memory_prompt_tokens > 0
    assert record.prompt_tokens == record.answer_prompt_tokens + record.memory_prompt_tokens
    assert "stale_summary" not in record.failure_modes
    assert "forgotten_fact" in record.failure_modes
    old = next(fact for fact in dialogue.planted_facts if fact.supersedes is None)
    assert old.text in fake.prompts[0][-1].content


def test_embedding_usage_is_added_to_the_probe_memory_cost(
    dialogues: dict[str, EvalDialogue],
) -> None:
    dialogue = dialogues["dlg-01-cor-imediata"]
    settings = load_settings({})
    accounting = EmbedAccounting()
    embedder = _CountingEmbedder(accounting)
    client = EstimatingChatClient(FakeLLM("não sei"), model="fake", settings=settings)

    records = run_evaluation(
        [dialogue],
        ["vector"],
        settings,
        client=client,
        embedder=embedder,
        embed_accounting=accounting,
    )

    assert records[0].memory_prompt_tokens == embedder.tokens
    assert records[0].memory_prompt_tokens > 0
    assert accounting.consume() == (0, 0.0)


def test_runner_propagates_rate_limit_when_retries_are_exhausted(
    dialogues: dict[str, EvalDialogue],
) -> None:
    client = RetryingChatClient(
        _AlwaysLimited(),
        max_attempts=2,
        base_delay_s=0,
        max_delay_s=0,
        sleep=lambda _delay: None,
    )

    with pytest.raises(TransientLLMError):
        run_evaluation(
            [dialogues["dlg-01-cor-imediata"]],
            ["window"],
            load_settings({}),
            client=client,
        )


def test_offline_run_over_the_shipped_dataset_stays_inside_the_harness_contract() -> None:
    settings = load_settings({})
    client = EstimatingChatClient(FakeLLM("resposta-fake"), model="fake", settings=settings)
    dialogues = load_eval_dialogues()

    records = run_evaluation(dialogues, ["window", "summary", "vector"], settings, client=client)

    assert len(records) == len(dialogues) * 3
    assert {record.strategy for record in records} == {"window", "summary", "vector"}
    assert all(0 <= record.fact_recall <= 1 for record in records)
    assert all(set(record.failure_modes) <= set(FAILURE_MODES) for record in records)
    assert all(record.correct is False and record.graded_by == "none" for record in records)
    assert all(record.cached is False and record.estimated_cost_usd == 0 for record in records)
    assert all(record.prompt_tokens > 0 and record.latency_ms >= 0 for record in records)
    window = [record for record in records if record.strategy == "window"]
    assert all(
        record.fact_recall == 1 and "forgotten_fact" not in record.failure_modes
        for record in window
        if record.probe_type == "immediate_recall"
    )
    assert all(
        "retrieved_distractor" in record.failure_modes and record.fact_recall == 1
        for record in window
        if record.probe_type == "distractor"
    )
    assert all("stale_summary" not in record.failure_modes for record in window)
    summary_updates = [
        record
        for record in records
        if record.strategy == "summary" and record.probe_type == "fact_update"
    ]
    assert summary_updates
    assert all(
        record.fact_recall == 1 and "stale_summary" not in record.failure_modes
        for record in summary_updates
    )


def test_cli_writes_markdown_and_csv_and_reuses_the_disk_cache(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    first = tmp_path / "first"
    second = tmp_path / "second"
    args = [
        "--strategy",
        "window",
        "--limit",
        "1",
        "--cache-dir",
        str(cache),
    ]

    assert main([*args, "--output-dir", str(first)], environ={}) == 0
    assert main([*args, "--output-dir", str(second)], environ={}) == 0
    first_record = json.loads((first / "records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    second_record = json.loads(
        (second / "records.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    table = list(csv.DictReader((first / "summary.csv").open(encoding="utf-8")))
    markdown = (first / "summary.md").read_text(encoding="utf-8")

    assert first_record["cached"] is False
    assert second_record["cached"] is True
    assert second_record["answer"] == "resposta-fake"
    assert any(row["tipo"] == "all" and int(row["n"]) == 1 for row in table)
    assert "Não é resultado publicado do estudo." in markdown
    assert list(cache.glob("*.json"))


def test_cli_rejects_a_bad_strategy_and_a_non_positive_limit(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--limit", "0", "--strategy", "window", "--output-dir", str(tmp_path)])

    code = main(["--strategy", "memoria", "--output-dir", str(tmp_path)], environ={})

    assert code == 1


class _Run:
    def __init__(self, record: object, prompts: list[tuple[Message, ...]]) -> None:
        self.record = record
        self.prompts = prompts


def _run(
    dialogue: EvalDialogue,
    strategy: str,
    settings: object,
    *,
    reply: str,
    clock: _Clock | None = None,
) -> _Run:
    fake = FakeLLM(reply)
    client: object = EstimatingChatClient(fake, model="fake", settings=settings)  # type: ignore[arg-type]
    if clock is not None:
        client = _Clocked(client, clock, 0.05)
    records = run_evaluation(
        [dialogue],
        [strategy],
        settings,  # type: ignore[arg-type]
        client=client,  # type: ignore[arg-type]
        clock=clock,
    )
    return _Run(records[0], fake.prompts)


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _Clocked:
    def __init__(self, inner: object, clock: _Clock, seconds: float) -> None:
        self.inner = inner
        self.clock = clock
        self.seconds = seconds
        self.model = inner.model  # type: ignore[attr-defined]

    def complete(self, messages: Sequence[Message]) -> Completion:
        self.clock.advance(self.seconds)
        return self.inner.complete(messages)  # type: ignore[attr-defined]


class _CountingEmbedder:
    def __init__(self, accounting: EmbedAccounting) -> None:
        self.accounting = accounting
        self.tokens = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        used = sum(max(len(text), 1) for text in texts)
        self.tokens += used
        self.accounting.add(used, 0.0)
        return [[float(len(text)), 0.0] for text in texts]


class _AlwaysLimited:
    model = "fake"

    def complete(self, messages: Sequence[Message]) -> Completion:
        raise TransientLLMError("limited")
