"""Reproduz o diálogo scriptado e chama o modelo só na sondagem.

O par usuário/assistente do dataset entra na memória como foi escrito.
A resposta do modelo não substitui o turno seguinte: senão o recall mediria
o texto gerado, não o fato plantado. A pergunta entra no prompt e só depois
é indexada, na mesma ordem do `Agent.step`.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from agent_memory_study.config import Settings
from agent_memory_study.embeddings import Embedder
from agent_memory_study.eval_dialogues import EvalDialogue, EvalProbe
from agent_memory_study.factory import STRATEGIES, build_memory
from agent_memory_study.llm_runtime import ChatClient, EmbedAccounting
from agent_memory_study.memory.summary import FakeSummarizer, LLMSummarizer, Summarizer
from agent_memory_study.metrics import Judge, ProbeRecord, locate_probe_turn, score_probe
from agent_memory_study.types import Message
from agent_memory_study.usage import Completion


@dataclass
class _Meter:
    inner: ChatClient
    calls: list[Completion]

    @property
    def model(self) -> str:
        return self.inner.model

    def complete(self, messages: Sequence[Message]) -> Completion:
        result = self.inner.complete(messages)
        if not isinstance(result, Completion):
            raise TypeError("chat client must return Completion")
        self.calls.append(result)
        return result


def run_evaluation(
    dialogues: Sequence[EvalDialogue],
    strategies: Sequence[str],
    settings: Settings,
    *,
    client: ChatClient,
    judge: Judge | None = None,
    embedder: Embedder | None = None,
    summarizer: Summarizer | None = None,
    summarize_with_client: bool = False,
    embed_accounting: EmbedAccounting | None = None,
    clock: Callable[[], float] | None = None,
) -> list[ProbeRecord]:
    """Joga cada diálogo em cada estratégia e devolve um registro por sondagem."""

    chosen = _check_strategies(strategies)
    now = clock if clock is not None else time.perf_counter
    meter = _Meter(inner=client, calls=[])
    active_summarizer = summarizer
    if active_summarizer is None:
        active_summarizer = LLMSummarizer(meter) if summarize_with_client else FakeSummarizer()
    records: list[ProbeRecord] = []
    for strategy in chosen:
        for dialogue in dialogues:
            records.extend(
                _play(
                    dialogue,
                    strategy,
                    settings,
                    meter=meter,
                    judge=judge,
                    embedder=embedder if strategy == "vector" else None,
                    summarizer=active_summarizer if strategy == "summary" else None,
                    accounting=embed_accounting if strategy == "vector" else None,
                    now=now,
                )
            )
    return records


def _play(
    dialogue: EvalDialogue,
    strategy: str,
    settings: Settings,
    *,
    meter: _Meter,
    judge: Judge | None,
    embedder: Embedder | None,
    summarizer: Summarizer | None,
    accounting: EmbedAccounting | None,
    now: Callable[[], float],
) -> list[ProbeRecord]:
    memory = build_memory(
        strategy,
        settings,
        summarizer=summarizer,
        embedder=embedder,
    )
    by_turn: dict[int, list[EvalProbe]] = {}
    for probe in dialogue.probes:
        by_turn.setdefault(locate_probe_turn(dialogue, probe), []).append(probe)
    seen = len(meter.calls)
    records: list[ProbeRecord] = []

    def drain() -> list[Completion]:
        nonlocal seen
        fresh = meter.calls[seen:]
        seen = len(meter.calls)
        return fresh

    for index, turn in enumerate(dialogue.turns):
        probes = by_turn.get(index)
        if not probes:
            memory.add(turn)
            continue
        if turn.role != "user":
            raise ValueError(f"{dialogue.id}: probe turn {index} is not a user turn")
        context = list(memory.context(query=turn.content))
        for probe_index, probe in enumerate(probes):
            if probe_index == 0:
                memory_calls = drain()
                embed_tokens, embed_cost = _consume(accounting)
            else:
                memory_calls = []
                embed_tokens, embed_cost = (0, 0.0)
            prompt = (*context, Message("user", probe.question))
            start = now()
            meter.complete(prompt)
            latency_ms = (now() - start) * 1000.0
            answer_calls = drain()
            if len(answer_calls) != 1:
                raise RuntimeError("probe answer must be a single completion")
            memory_prompt, memory_completion, memory_cost = _sum_calls(memory_calls)
            records.append(
                score_probe(
                    dialogue,
                    probe,
                    context,
                    strategy=strategy,
                    answer_completion=answer_calls[0],
                    latency_ms=latency_ms,
                    memory_prompt_tokens=memory_prompt + embed_tokens,
                    memory_completion_tokens=memory_completion,
                    memory_cost_usd=memory_cost + embed_cost,
                    judge=judge,
                )
            )
        memory.add(turn)

    records = _fold_trailing(records, drain(), _consume(accounting))
    return records


def _fold_trailing(
    records: list[ProbeRecord],
    calls: Sequence[Completion],
    embed_usage: tuple[int, float],
) -> list[ProbeRecord]:
    # Indexar a pergunta da última sondagem acontece depois da resposta.
    # O custo dessa escrita entra no mesmo registro, senão a execução o perde.
    if not records:
        return records
    embed_tokens, embed_cost = embed_usage
    prompt_tokens, completion_tokens, cost = _sum_calls(calls)
    prompt_tokens += embed_tokens
    cost += embed_cost
    if prompt_tokens == 0 and completion_tokens == 0 and cost == 0:
        return records
    last = records[-1]
    records[-1] = replace(
        last,
        prompt_tokens=last.prompt_tokens + prompt_tokens,
        completion_tokens=last.completion_tokens + completion_tokens,
        memory_prompt_tokens=last.memory_prompt_tokens + prompt_tokens,
        memory_completion_tokens=last.memory_completion_tokens + completion_tokens,
        estimated_cost_usd=last.estimated_cost_usd + cost,
    )
    return records


def _sum_calls(calls: Sequence[Completion]) -> tuple[int, int, float]:
    return (
        sum(call.prompt_tokens for call in calls),
        sum(call.completion_tokens for call in calls),
        sum(call.estimated_cost_usd for call in calls),
    )


def _consume(accounting: EmbedAccounting | None) -> tuple[int, float]:
    if accounting is None:
        return (0, 0.0)
    return accounting.consume()


def _check_strategies(strategies: Sequence[str]) -> tuple[str, ...]:
    if not strategies:
        raise ValueError("at least one memory strategy is required")
    unknown = [name for name in strategies if name not in STRATEGIES]
    if unknown:
        allowed = ", ".join(STRATEGIES)
        raise ValueError(f"unknown memory strategy: {unknown[0]!r}; expected one of: {allowed}")
    return tuple(strategies)
