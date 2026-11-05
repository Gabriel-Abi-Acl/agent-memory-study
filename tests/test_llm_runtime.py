"""Contrato de retry, cache e cliente HTTP: nenhuma chamada sai desta máquina."""

from __future__ import annotations

import json
import threading
from collections.abc import Mapping
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from agent_memory_study.config import load_settings
from agent_memory_study.llm_runtime import (
    CachingChatClient,
    DiskCache,
    EmbedAccounting,
    HttpResponse,
    LLMError,
    OpenAIChatClient,
    RetryingChatClient,
    TransientLLMError,
    build_chat_client,
    build_embedder,
    cache_key,
    urllib_transport,
)
from agent_memory_study.types import Message
from agent_memory_study.usage import Completion


def test_retry_uses_backoff_and_stops_on_a_permanent_error() -> None:
    sleeps: list[float] = []
    flaky = _Flaky(failures=[TransientLLMError("limited"), TransientLLMError("still")])
    client = RetryingChatClient(
        flaky,
        max_attempts=4,
        base_delay_s=0.25,
        max_delay_s=8,
        sleep=sleeps.append,
    )

    assert client.complete([Message("user", "oi")]).text == "ok"
    assert sleeps == [0.25, 0.5]
    assert flaky.calls == 3

    permanent = _Flaky(failures=[LLMError("400 bad")])
    sleeps.clear()
    wrapped = RetryingChatClient(
        permanent,
        max_attempts=3,
        base_delay_s=0.25,
        max_delay_s=8,
        sleep=sleeps.append,
    )
    with pytest.raises(LLMError, match="400"):
        wrapped.complete([Message("user", "oi")])
    assert permanent.calls == 1
    assert sleeps == []


def test_retry_after_is_capped_and_the_last_failure_does_not_sleep() -> None:
    sleeps: list[float] = []
    capped = _Flaky(failures=[TransientLLMError("limited", retry_after_s=100)])
    client = RetryingChatClient(
        capped,
        max_attempts=3,
        base_delay_s=0.25,
        max_delay_s=8,
        sleep=sleeps.append,
    )

    assert client.complete([Message("user", "oi")]).text == "ok"
    assert sleeps == [8]

    always = _AlwaysLimited()
    sleeps.clear()
    exhausted = RetryingChatClient(
        always,
        max_attempts=2,
        base_delay_s=0.5,
        max_delay_s=8,
        sleep=sleeps.append,
    )
    with pytest.raises(TransientLLMError):
        exhausted.complete([Message("user", "oi")])
    assert sleeps == [0.5]
    assert always.calls == 2


def test_disk_cache_round_trips_and_rejects_a_path_like_key(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    key = cache_key("chat", "fake", [Message("user", "oi")])
    other = cache_key("chat", "fake", [Message("user", "tchau")])

    assert cache.get(key) is None
    cache.put(key, {"text": "ok", "prompt_tokens": 1, "completion_tokens": 1})
    assert cache.get(key) == {"text": "ok", "prompt_tokens": 1, "completion_tokens": 1}
    assert other != key
    (tmp_path / f"{key}.json").write_text("{", encoding="utf-8")
    assert cache.get(key) is None
    with pytest.raises(ValueError, match="sha256"):
        cache.get("../secrets")


def test_openai_chat_retries_429_then_prices_the_successful_call() -> None:
    key = "sk-test-secret"
    settings = _openai_settings(key)
    body = _chat_body("ocre", prompt_tokens=1000, completion_tokens=500)
    transport = _QueueTransport(
        [
            HttpResponse(429, b"slow down", {"Retry-After": "1.5"}),
            HttpResponse(500, b"later", {}),
            HttpResponse(200, body, {}),
        ]
    )
    sleeps: list[float] = []
    client = RetryingChatClient(
        OpenAIChatClient(settings, transport=transport),
        max_attempts=4,
        base_delay_s=0.25,
        max_delay_s=8,
        sleep=sleeps.append,
    )

    completion = client.complete([Message("user", "Qual é a cor?")])

    assert completion.text == "ocre"
    assert completion.prompt_tokens == 1000
    assert completion.completion_tokens == 500
    assert completion.estimated_cost_usd == pytest.approx(0.00045)
    assert completion.cached is False
    assert sleeps == [1.5, 0.5]
    assert transport.calls[0][1].endswith("/chat/completions")
    assert transport.calls[0][2]["Authorization"] == f"Bearer {key}"
    assert key.encode() not in transport.calls[0][3]
    sent = json.loads(transport.calls[0][3])
    assert sent["temperature"] == 0
    assert sent["model"] == "gpt-4o-mini"


def test_openai_error_redacts_the_key_and_does_not_retry_a_400() -> None:
    key = "sk-test-secret"
    settings = _openai_settings(key)
    transport = _QueueTransport([HttpResponse(400, f"invalid {key}".encode(), {})])
    sleeps: list[float] = []
    client = RetryingChatClient(
        OpenAIChatClient(settings, transport=transport),
        max_attempts=3,
        base_delay_s=0.25,
        max_delay_s=8,
        sleep=sleeps.append,
    )

    with pytest.raises(LLMError, match="HTTP 400") as exc_info:
        client.complete([Message("user", "oi")])

    assert key not in str(exc_info.value)
    assert "[redacted]" in str(exc_info.value)
    assert len(transport.calls) == 1
    assert sleeps == []


def test_disk_cache_skips_the_second_call_and_zeroes_its_cost(tmp_path: Path) -> None:
    key = "sk-test-secret"
    settings = _openai_settings(key)
    transport = _QueueTransport(
        [HttpResponse(200, _chat_body("ok", prompt_tokens=1000, completion_tokens=500), {})]
    )
    client = CachingChatClient(
        OpenAIChatClient(settings, transport=transport),
        DiskCache(tmp_path),
    )

    first = client.complete([Message("user", "oi")])
    second = client.complete([Message("user", "oi")])
    stored = "\n".join(path.read_text(encoding="utf-8") for path in tmp_path.glob("*.json"))

    assert first.cached is False
    assert first.estimated_cost_usd == pytest.approx(0.00045)
    assert second.cached is True
    assert second.text == "ok"
    assert second.prompt_tokens == 1000
    assert second.completion_tokens == 500
    assert second.estimated_cost_usd == 0
    assert len(transport.calls) == 1
    assert key not in stored


def test_embeddings_come_back_in_input_order_and_update_the_accounting() -> None:
    settings = _openai_settings("sk-test-secret")
    payload = {
        "data": [
            {"index": 1, "embedding": [0.0, 1.0]},
            {"index": 0, "embedding": [1.0, 0.0]},
        ],
        "usage": {"prompt_tokens": 11},
    }
    transport = _QueueTransport(
        [
            HttpResponse(429, b"slow", {"Retry-After": "0"}),
            HttpResponse(200, json.dumps(payload).encode(), {}),
        ]
    )
    sleeps: list[float] = []
    accounting = EmbedAccounting()
    embedder = build_embedder(
        settings,
        accounting=accounting,
        transport=transport,
        sleep=sleeps.append,
    )

    assert embedder.embed(["um", "dois"]) == [[1.0, 0.0], [0.0, 1.0]]
    tokens, cost = accounting.consume()
    assert tokens == 11
    assert cost == pytest.approx(11 / 1_000_000 * 0.02)
    assert sleeps == [0]
    assert transport.calls[-1][1].endswith("/embeddings")
    assert accounting.consume() == (0, 0.0)


def test_openai_builder_refuses_to_start_without_a_key() -> None:
    settings = load_settings({"LLM_PROVIDER": "openai", "EMBEDDING_PROVIDER": "openai"})
    transport = _QueueTransport([])

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_chat_client(settings, transport=transport)
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_embedder(settings, transport=transport)
    assert transport.calls == []


def test_urllib_transport_does_not_follow_redirects() -> None:
    seen: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            seen.append(self.path)
            self.send_response(302)
            self.send_header("Location", "http://127.0.0.1:9/steal")
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address
        response = urllib_transport(
            "POST",
            f"http://{host}:{port}/v1/chat/completions",
            {"Authorization": "Bearer sk-test-secret", "Content-Type": "application/json"},
            b"{}",
            2.0,
        )
    finally:
        server.shutdown()
        server.server_close()

    assert response.status == 302
    assert seen == ["/v1/chat/completions"]


class _Flaky:
    model = "fake"

    def __init__(self, failures: list[Exception]) -> None:
        self._failures = list(failures)
        self.calls = 0

    def complete(self, messages: list[Message]) -> Completion:
        self.calls += 1
        if self._failures:
            raise self._failures.pop(0)
        return Completion("ok", 1, 1, 0.0, False, self.model)


class _AlwaysLimited:
    model = "fake"

    def __init__(self) -> None:
        self.calls = 0

    def complete(self, messages: list[Message]) -> Completion:
        self.calls += 1
        raise TransientLLMError("limited")


class _QueueTransport:
    def __init__(self, responses: list[HttpResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, Mapping[str, str], bytes, float]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        self.calls.append((method, url, dict(headers), body, timeout))
        if not self._responses:
            raise AssertionError("transport has no queued response")
        return self._responses.pop(0)


def _openai_settings(api_key: str):
    return load_settings(
        {
            "LLM_PROVIDER": "openai",
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": api_key,
            "OPENAI_MODEL": "gpt-4o-mini",
            "EMBEDDING_MODEL": "text-embedding-3-small",
            "LLM_TIMEOUT_S": "5",
        }
    )


def _chat_body(text: str, *, prompt_tokens: int, completion_tokens: int) -> bytes:
    return json.dumps(
        {
            "choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        }
    ).encode()
