"""Contrato: sem variável de provedor, o estudo fica offline e não exige chave."""

import pytest

from agent_memory_study.config import load_settings


def test_settings_default_to_offline_fakes_without_an_api_key() -> None:
    settings = load_settings({})

    assert settings.llm_provider == "fake"
    assert settings.embedding_provider == "fake"
    assert settings.openai_api_key is None
    assert "openai_api_key=<empty>" in repr(settings)


def test_settings_repr_does_not_include_the_api_key() -> None:
    settings = load_settings({"OPENAI_API_KEY": "sk-test-secret"})

    assert settings.openai_api_key == "sk-test-secret"
    assert "sk-test-secret" not in repr(settings)
    assert "openai_api_key=<set>" in repr(settings)


def test_settings_reject_a_provider_that_would_call_the_network() -> None:
    with pytest.raises(ValueError, match="not implemented"):
        load_settings({"LLM_PROVIDER": "openai", "OPENAI_API_KEY": "sk-test-secret"})
