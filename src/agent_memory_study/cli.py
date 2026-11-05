"""Linha de comando do harness. Default offline; chave só se o ambiente tiver."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from agent_memory_study.config import load_settings
from agent_memory_study.eval_dialogues import load_eval_dialogues
from agent_memory_study.factory import STRATEGIES
from agent_memory_study.llm_runtime import (
    DiskCache,
    EmbedAccounting,
    LLMError,
    build_chat_client,
    build_embedder,
)
from agent_memory_study.metrics import LLMJudge, aggregate, write_reports
from agent_memory_study.runner import run_evaluation


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None) -> int:
    args = _parser().parse_args(None if argv is None else list(argv))
    env = os.environ if environ is None else environ
    try:
        settings = load_settings(env)
        strategies = _strategies(args.strategy)
        dialogues = load_eval_dialogues(args.dialogues)
        if args.limit is not None:
            dialogues = dialogues[: args.limit]
        cache = _cache(args.cache_dir, env)
        client = build_chat_client(settings, cache=cache, fake_reply=args.fake_reply)
        judge = LLMJudge(client) if args.judge else None
        accounting = EmbedAccounting()
        embedder = None
        if "vector" in strategies:
            embedder = build_embedder(settings, accounting=accounting)
        records = run_evaluation(
            dialogues,
            strategies,
            settings,
            client=client,
            judge=judge,
            embedder=embedder,
            summarize_with_client=settings.llm_provider == "openai",
            embed_accounting=accounting if embedder is not None else None,
        )
        rows = aggregate(records)
        model = "fake" if settings.llm_provider == "fake" else settings.openai_model
        write_reports(
            args.output_dir,
            records,
            rows,
            provider=settings.llm_provider,
            model=model,
            dialogue_count=len(dialogues),
            strategies=strategies,
            judge=args.judge,
        )
    except (LLMError, ValueError) as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1
    print(args.output_dir / "summary.md")
    return 0


def cli(argv: Sequence[str] | None = None) -> None:
    raise SystemExit(main(argv))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="evaluate-memory",
        description=(
            "Roda cada diálogo do dataset v1 numa estratégia de memória e grava "
            "métricas por sondagem. O provedor padrão é fake, sem rede. "
            "O arquivo gerado é o traço desta execução, não o resultado do estudo."
        ),
    )
    parser.add_argument(
        "--strategy",
        action="append",
        required=True,
        help="window, summary, vector ou all. Aceita lista separada por vírgula e repetição.",
    )
    parser.add_argument(
        "--dialogues",
        type=Path,
        default=None,
        help="JSONL v1. O default é data/eval/dialogues.v1.jsonl.",
    )
    parser.add_argument("--limit", type=_positive_int, default=None, help="Primeiros N diálogos.")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Onde gravar csv, md e jsonl."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache de respostas. Se omitido, usa LLM_CACHE_DIR quando essa variável existe.",
    )
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Pede ao mesmo modelo para aceitar paráfrase quando a normalização falha.",
    )
    parser.add_argument(
        "--fake-reply",
        default="resposta-fake",
        help="Texto do LLM fake. Ignorado quando LLM_PROVIDER=openai.",
    )
    return parser


def _strategies(raw: Sequence[str]) -> tuple[str, ...]:
    names: list[str] = []
    for item in raw:
        for part in item.split(","):
            name = part.strip()
            if not name:
                continue
            if name == "all":
                for strategy in STRATEGIES:
                    if strategy not in names:
                        names.append(strategy)
                continue
            if name not in STRATEGIES:
                allowed = ", ".join(STRATEGIES)
                raise ValueError(f"unknown memory strategy: {name!r}; expected one of: {allowed}")
            if name not in names:
                names.append(name)
    if not names:
        raise ValueError("at least one memory strategy is required")
    return tuple(names)


def _cache(flag: Path | None, env: Mapping[str, str]) -> DiskCache | None:
    if flag is not None:
        return DiskCache(flag)
    raw = env.get("LLM_CACHE_DIR", "").strip()
    if not raw:
        return None
    return DiskCache(Path(raw))


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed
