"""Cliente de chat com retry, cache em disco e chamada OpenAI opcional.

O transporte HTTP é injetável. Teste nenhum precisa de chave nem de rede.
Redirect é recusado: seguir o Location reenviaria o Bearer para outro host.
`HTTP(S)_PROXY` do ambiente é ignorado pelo mesmo motivo; quem precisa de
gateway aponta `OPENAI_BASE_URL` para ele.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TypeVar

from agent_memory_study.config import Settings
from agent_memory_study.embeddings import Embedder, FakeEmbedder
from agent_memory_study.llm import FakeLLM
from agent_memory_study.types import Message
from agent_memory_study.usage import (
    Completion,
    estimate_cost_usd,
    estimate_message_tokens,
    estimate_tokens,
)

_RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})
_HEX_KEY = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class LLMError(RuntimeError):
    """Falha permanente. A mensagem não carrega a chave."""


class TransientLLMError(LLMError):
    """429, 5xx ou queda de rede. O chamador pode tentar de novo."""

    def __init__(self, message: str, *, retry_after_s: float | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ChatClient(Protocol):
    model: str

    def complete(self, messages: Sequence[Message]) -> Completion:
        """Gera a fala e devolve tokens e custo estimado."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: Mapping[str, str]


class Transport(Protocol):
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes,
        timeout: float,
    ) -> HttpResponse:
        """Executa um POST e devolve status, corpo e cabeçalhos."""


class EmbedAccounting:
    """Soma tokens e custo de embeddings desde o último consume.

    O embedder real incrementa isto. O falso não mexe, e o consume segue zero.
    """

    def __init__(self) -> None:
        self.prompt_tokens = 0
        self.cost_usd = 0.0

    def add(self, prompt_tokens: int, cost_usd: float) -> None:
        if prompt_tokens < 0 or cost_usd < 0:
            raise ValueError("embedding usage must be >= 0")
        self.prompt_tokens += prompt_tokens
        self.cost_usd += cost_usd

    def consume(self) -> tuple[int, float]:
        snapshot = (self.prompt_tokens, self.cost_usd)
        self.prompt_tokens = 0
        self.cost_usd = 0.0
        return snapshot


def call_with_retry(
    operation: Callable[[], _T],
    *,
    max_attempts: int,
    base_delay_s: float,
    max_delay_s: float,
    sleep: Callable[[float], None],
) -> _T:
    """Repete só erro transitório. A última falha sobe sem dormir de novo."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")
    for attempt in range(1, max_attempts + 1):
        try:
            return operation()
        except TransientLLMError as exc:
            if attempt >= max_attempts:
                raise
            sleep(_delay(exc, attempt, base_delay_s, max_delay_s))
    raise AssertionError("retry loop ended without a result")


class RetryingChatClient:
    """Aplica a política de retry em volta de outro cliente de chat."""

    def __init__(
        self,
        inner: ChatClient,
        *,
        max_attempts: int,
        base_delay_s: float,
        max_delay_s: float,
        sleep: Callable[[float], None],
    ) -> None:
        self.inner = inner
        self.model = inner.model
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep

    def complete(self, messages: Sequence[Message]) -> Completion:
        return call_with_retry(
            lambda: self.inner.complete(messages),
            max_attempts=self._max_attempts,
            base_delay_s=self._base_delay_s,
            max_delay_s=self._max_delay_s,
            sleep=self._sleep,
        )


class RetryingEmbedder:
    """Mesma política de retry, no embedder."""

    def __init__(
        self,
        inner: Embedder,
        *,
        max_attempts: int,
        base_delay_s: float,
        max_delay_s: float,
        sleep: Callable[[float], None],
    ) -> None:
        self.inner = inner
        self._max_attempts = max_attempts
        self._base_delay_s = base_delay_s
        self._max_delay_s = max_delay_s
        self._sleep = sleep

    def embed(self, texts: list[str]) -> list[list[float]]:
        return call_with_retry(
            lambda: self.inner.embed(texts),
            max_attempts=self._max_attempts,
            base_delay_s=self._base_delay_s,
            max_delay_s=self._max_delay_s,
            sleep=self._sleep,
        )


class DiskCache:
    """JSON por chave sha256. A chave da API não entra no nome nem no corpo."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict):
            return None
        return payload

    def put(self, key: str, payload: dict[str, Any]) -> None:
        path = self._path(key)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _path(self, key: str) -> Path:
        if _HEX_KEY.fullmatch(key) is None:
            raise ValueError("cache key must be a sha256 hex digest")
        return self.directory / f"{key}.json"


class CachingChatClient:
    """Guarda o texto e os tokens. Hit não chama o cliente interno e custa zero."""

    def __init__(self, inner: ChatClient, cache: DiskCache) -> None:
        self.inner = inner
        self.cache = cache
        self.model = inner.model

    def complete(self, messages: Sequence[Message]) -> Completion:
        key = cache_key("chat", self.model, messages)
        hit = self.cache.get(key)
        if hit is not None and _cached_completion_fields(hit):
            return Completion(
                text=str(hit["text"]),
                prompt_tokens=int(hit["prompt_tokens"]),
                completion_tokens=int(hit["completion_tokens"]),
                estimated_cost_usd=0.0,
                cached=True,
                model=self.model,
            )
        result = self.inner.complete(messages)
        self.cache.put(
            key,
            {
                "text": result.text,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
            },
        )
        return result


class EstimatingChatClient:
    """Adapta um `LLM` que só devolve texto para a contagem do harness."""

    def __init__(self, inner: FakeLLM, *, model: str, settings: Settings) -> None:
        self.inner = inner
        self.model = model
        self._settings = settings

    def complete(self, messages: Sequence[Message]) -> Completion:
        text = self.inner.complete(messages)
        prompt_tokens = estimate_message_tokens(messages)
        completion_tokens = estimate_tokens(text)
        return Completion(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=estimate_cost_usd(
                self._settings,
                self.model,
                prompt_tokens,
                completion_tokens,
                cached=False,
            ),
            cached=False,
            model=self.model,
        )


class OpenAIChatClient:
    """Chat Completions. Sem chave, falha antes de abrir o socket."""

    def __init__(self, settings: Settings, *, transport: Transport) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        self.model = settings.openai_model
        self._settings = settings
        self._poster = _OpenAIPost(settings, transport, settings.openai_api_key)

    def complete(self, messages: Sequence[Message]) -> Completion:
        payload = self._poster.post(
            "/chat/completions",
            {
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": message.role, "content": message.content} for message in messages
                ],
            },
        )
        return _parse_chat(payload, messages, self._settings, self.model)


class OpenAIEmbedder:
    """Embeddings. O uso entra na contabilidade compartilhada com o runner."""

    def __init__(
        self,
        settings: Settings,
        *,
        transport: Transport,
        accounting: EmbedAccounting | None = None,
    ) -> None:
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        self._settings = settings
        self._model = settings.embedding_model
        self._accounting = accounting
        self._poster = _OpenAIPost(settings, transport, settings.openai_api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = self._poster.post(
            "/embeddings",
            {"model": self._model, "input": texts},
        )
        vectors, reported = _parse_embeddings(payload, len(texts))
        prompt_tokens = reported
        if prompt_tokens is None:
            prompt_tokens = sum(estimate_tokens(text) for text in texts)
        cost = estimate_cost_usd(
            self._settings,
            self._model,
            prompt_tokens,
            0,
            cached=False,
        )
        if self._accounting is not None:
            self._accounting.add(prompt_tokens, cost)
        return vectors


class _OpenAIPost:
    def __init__(self, settings: Settings, transport: Transport, api_key: str) -> None:
        self._settings = settings
        self._transport = transport
        self._api_key = api_key

    def post(self, path: str, payload: dict[str, Any]) -> Any:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "agent-memory-study",
        }
        try:
            response = self._transport(
                "POST",
                f"{self._settings.openai_base_url}{path}",
                headers,
                body,
                self._settings.llm_timeout_s,
            )
        except TransientLLMError as exc:
            raise TransientLLMError(_redact(str(exc), self._api_key)) from exc
        if response.status != 200:
            _raise_for_status(response, self._api_key)
        try:
            return json.loads(response.body.decode("utf-8"))
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise LLMError("OpenAI response was not JSON") from exc


def build_chat_client(
    settings: Settings,
    *,
    cache: DiskCache | None = None,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
    fake_reply: str = "resposta-fake",
) -> ChatClient:
    """Monta fake ou OpenAI, sempre com retry. Cache só se o diretório existir."""

    if settings.llm_provider == "fake":
        inner: ChatClient = EstimatingChatClient(
            FakeLLM(fake_reply),
            model="fake",
            settings=settings,
        )
    elif settings.llm_provider == "openai":
        inner = OpenAIChatClient(settings, transport=transport or urllib_transport)
    else:
        raise ValueError(f"unknown LLM provider: {settings.llm_provider!r}")
    retrying = RetryingChatClient(
        inner,
        max_attempts=settings.llm_max_attempts,
        base_delay_s=settings.llm_retry_base_delay_s,
        max_delay_s=settings.llm_retry_max_delay_s,
        sleep=sleep,
    )
    if cache is None:
        return retrying
    return CachingChatClient(retrying, cache)


def build_embedder(
    settings: Settings,
    *,
    accounting: EmbedAccounting | None = None,
    transport: Transport | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Embedder:
    """Fake não conta token. OpenAI soma na contabilidade e repete 429."""

    if settings.embedding_provider == "fake":
        return FakeEmbedder()
    if settings.embedding_provider != "openai":
        raise ValueError(f"unknown embedding provider: {settings.embedding_provider!r}")
    inner = OpenAIEmbedder(
        settings,
        transport=transport or urllib_transport,
        accounting=accounting,
    )
    return RetryingEmbedder(
        inner,
        max_attempts=settings.llm_max_attempts,
        base_delay_s=settings.llm_retry_base_delay_s,
        max_delay_s=settings.llm_retry_max_delay_s,
        sleep=sleep,
    )


def cache_key(
    kind: str, model: str, messages: Sequence[Message], *, temperature: float = 0.0
) -> str:
    payload = {
        "kind": kind,
        "model": model,
        "temperature": temperature,
        "messages": [{"role": message.role, "content": message.content} for message in messages],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def urllib_transport(
    method: str,
    url: str,
    headers: Mapping[str, str],
    body: bytes,
    timeout: float,
) -> HttpResponse:
    """POST sem seguir redirect e sem proxy do ambiente."""

    request = urllib.request.Request(url, data=body, headers=dict(headers), method=method)
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            return HttpResponse(
                status=response.status,
                body=response.read(),
                headers={key: value for key, value in response.headers.items()},
            )
    except urllib.error.HTTPError as exc:
        payload = exc.read() if exc.fp is not None else b""
        header_map = {key: value for key, value in exc.headers.items()} if exc.headers else {}
        return HttpResponse(status=exc.code, body=payload, headers=header_map)
    except (urllib.error.URLError, TimeoutError) as exc:
        raise TransientLLMError(f"OpenAI transport failed: {exc.__class__.__name__}") from exc


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, msg, headers, fp)


def _delay(
    exc: TransientLLMError,
    attempt: int,
    base_delay_s: float,
    max_delay_s: float,
) -> float:
    if exc.retry_after_s is not None and exc.retry_after_s >= 0:
        raw = exc.retry_after_s
    else:
        raw = base_delay_s * (2 ** (attempt - 1))
    return min(max_delay_s, max(0.0, raw))


def _raise_for_status(response: HttpResponse, secret: str) -> None:
    detail = _redact(_snippet(response.body), secret)
    message = f"OpenAI HTTP {response.status}: {detail}"
    if response.status in _RETRYABLE_STATUS:
        raise TransientLLMError(message, retry_after_s=_retry_after(response.headers))
    raise LLMError(message)


def _retry_after(headers: Mapping[str, str]) -> float | None:
    raw = None
    for key, value in headers.items():
        if key.lower() == "retry-after":
            raw = value.strip()
            break
    if raw is None:
        return None
    try:
        parsed = float(raw)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _snippet(body: bytes) -> str:
    return " ".join(body.decode("utf-8", errors="replace").split())[:300]


def _redact(text: str, secret: str) -> str:
    if secret and secret in text:
        return text.replace(secret, "[redacted]")
    return text


def _cached_completion_fields(payload: dict[str, Any]) -> bool:
    text = payload.get("text")
    prompt_tokens = payload.get("prompt_tokens")
    completion_tokens = payload.get("completion_tokens")
    return (
        isinstance(text, str)
        and isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and isinstance(completion_tokens, int)
        and not isinstance(completion_tokens, bool)
        and prompt_tokens >= 0
        and completion_tokens >= 0
    )


def _parse_chat(
    payload: Any,
    messages: Sequence[Message],
    settings: Settings,
    model: str,
) -> Completion:
    if not isinstance(payload, dict):
        raise LLMError("OpenAI response has no choices")
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMError("OpenAI response has no choices")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LLMError("OpenAI response has no message")
    text = _message_text(message.get("content"))
    prompt_tokens = _usage_int(payload.get("usage"), "prompt_tokens")
    completion_tokens = _usage_int(payload.get("usage"), "completion_tokens")
    if prompt_tokens is None:
        prompt_tokens = estimate_message_tokens(messages)
    if completion_tokens is None:
        completion_tokens = estimate_tokens(text)
    return Completion(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        estimated_cost_usd=estimate_cost_usd(
            settings,
            model,
            prompt_tokens,
            completion_tokens,
            cached=False,
        ),
        cached=False,
        model=model,
    )


def _parse_embeddings(payload: Any, expected: int) -> tuple[list[list[float]], int | None]:
    if not isinstance(payload, dict):
        raise LLMError("OpenAI embeddings response is malformed")
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != expected:
        raise LLMError("OpenAI embeddings response has the wrong size")
    rows: list[tuple[int, list[float]]] = []
    for item in data:
        if not isinstance(item, dict):
            raise LLMError("OpenAI embeddings response is malformed")
        index = item.get("index")
        vector = item.get("embedding")
        if isinstance(index, bool) or not isinstance(index, int):
            raise LLMError("OpenAI embeddings response is malformed")
        if not isinstance(vector, list) or not _is_vector(vector):
            raise LLMError("OpenAI embeddings response is malformed")
        rows.append((index, [float(value) for value in vector]))
    rows.sort(key=lambda pair: pair[0])
    if [index for index, _vector in rows] != list(range(expected)):
        raise LLMError("OpenAI embeddings response is missing an index")
    return [vector for _index, vector in rows], _usage_int(payload.get("usage"), "prompt_tokens")


def _is_vector(values: list[Any]) -> bool:
    if not values:
        return False
    return all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                chunks.append(part["text"])
            else:
                raise LLMError("OpenAI message content is not text")
        return "".join(chunks)
    raise LLMError("OpenAI message content is not text")


def _usage_int(usage: Any, key: str) -> int | None:
    if not isinstance(usage, dict):
        return None
    value = usage.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value
