"""Um turno: monta o prompt com a memória, chama o modelo e grava o par.

A fala atual do usuário entra no prompt mesmo quando a estratégia não a
devolve. Ela só é indexada depois da resposta, junto com a fala do assistente.
"""

from __future__ import annotations

from agent_memory_study.llm import LLM
from agent_memory_study.memory.base import Memory
from agent_memory_study.types import Message


class Agent:
    def __init__(self, memory: Memory, llm: LLM) -> None:
        self.memory = memory
        self.llm = llm

    def step(self, user_text: str) -> str:
        user_message = Message(role="user", content=user_text)
        prompt = [*self.memory.context(query=user_text), user_message]
        reply_text = self.llm.complete(prompt)
        self.memory.add(user_message)
        self.memory.add(Message(role="assistant", content=reply_text))
        return reply_text
