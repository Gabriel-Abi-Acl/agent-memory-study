"""Contrato: sem variável de provedor, o estudo fica offline e não exige chave."""

import pytest

from agent_memory_study.config import load_settings


def test_settings_default_to_offline_fakes_without_an_api_key() -> None:
    settings = load_settings({})

    assert settings.llm_provider == "fake"
    assert settings.embedding_provider == "fake"
    assert settings.openai_api_key is None
    assert settings.llm_max_attempts == 4
    assert settings.prompt_usd_per_million is None
    assert settings.completion_usd_per_million is None
    assert "openai_api_key=<empty>" in repr(settings)


def test_settings_repr_does_not_include_the_api_key() -> None:
    settings = load_settings({"OPENAI_API_KEY": "sk-test-secret"})

    assert settings.openai_api_key == "sk-test-secret"
    assert "sk-test-secret" not in repr(settings)
    assert "openai_api_key=<set>" in repr(settings)


def test_settings_accept_openai_as_a_provider_name_without_calling_the_network() -> None:
    settings = load_settings(
        {
            "LLM_PROVIDER": "openai",
            "EMBEDDING_PROVIDER": "openai",
            "OPENAI_API_KEY": "sk-test-secret",
        }
    )

    assert settings.llm_provider == "openai"
    assert settings.embedding_provider == "openai"
    assert settings.openai_api_key == "sk-test-secret"
    assert "sk-test-secret" not in repr(settings)


def test_settings_reject_an_unknown_provider() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        load_settings({"LLM_PROVIDER": "anthropic"})


def test_settings_reject_a_lone_price_override() -> None:
    with pytest.raises(ValueError, match="together"):
        load_settings({"PROMPT_USD_PER_MILLION": "1"})


def test_settings_reject_credentials_in_the_base_url() -> None:
    with pytest.raises(ValueError, match="OPENAI_BASE_URL"):
        load_settings({"OPENAI_BASE_URL": "https://user:pass@example.com/v1"})
