"""Contrato: o turno seguinte recebe, no prompt, o fato que a estratégia ainda guarda."""

from agent_memory_study.agent import Agent
from agent_memory_study.config import load_settings
from agent_memory_study.factory import build_memory
from agent_memory_study.llm import FakeLLM


def test_agent_passes_recalled_fact_to_the_llm_on_the_next_turn() -> None:
    settings = load_settings(
        {
            "LLM_PROVIDER": "fake",
            "EMBEDDING_PROVIDER": "fake",
            "WINDOW_SIZE": "8",
        }
    )
    llm = FakeLLM(reply="ack")
    agent = Agent(memory=build_memory("window", settings), llm=llm)

    agent.step("O código interno do projeto é Nimbus-42.")
    reply = agent.step("Qual é o código interno do projeto?")

    assert reply == "ack"
    second_prompt = llm.prompts[1]
    assert [message.content for message in second_prompt] == [
        "O código interno do projeto é Nimbus-42.",
        "ack",
        "Qual é o código interno do projeto?",
    ]
    assert agent.memory.context()[-1].content == "ack"
