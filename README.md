# Estudo de memória em agentes conversacionais

Comparação técnica de três estratégias de memória num agente de diálogo multi-turno: janela deslizante, resumo incremental e recuperação vetorial. Os diálogos são sintéticos. Este repositório ainda não tem resultados experimentais.

## Objetivo

Medir, no mesmo conjunto de diálogos, como cada estratégia se comporta quando uma pergunta tardia depende de um fato dito antes. As medidas previstas são:

- recall do fato plantado: o contexto entregue ao modelo contém o trecho esperado;
- correção da resposta, contra um gabarito;
- tokens do prompt e latência por turno;
- modo de falha: fato expulso da janela, resumo desatualizado, ou distrator recuperado no lugar do fato.

Nada disso foi medido ainda. Os números entram só depois dos experimentos do PR4, a partir de execuções reais.

## Hipótese

Ainda não testada. O desenho do dataset e das métricas parte de três expectativas:

1. A janela deslizante conserva o fato enquanto ele está entre as últimas N mensagens e o perde quando a mensagem é expulsa.
2. O resumo gasta menos tokens em diálogos longos e preserva o tema, mas pode descartar um valor literal (nome, código, cor) ou manter um valor antigo depois de uma correção.
3. A recuperação vetorial devolve o enunciado do fato quando a pergunta se aproxima dele no espaço do embedder, e pode devolver um trecho parecido que não contém o fato.

Se alguma dessas expectativas falhar na medição, o relatório deve dizer isso. Elas não são conclusão.

## Metodologia planejada

Unidade experimental: um diálogo sintético com fatos plantados e sondagens posteriores. Cada sondagem aponta os ids dos fatos de que a resposta depende. O texto esperado da resposta (gabarito) fica para o PR2; o fixture deste PR só fixa o trecho literal e o turno em que ele aparece.

Condições: `window`, `summary` e `vector`, no mesmo laço de agente. Tamanho da janela, limite do buffer de resumo e `top_k` vêm de variável de ambiente.

O que cada turno faz hoje:

1. a estratégia devolve o contexto que considera relevante para a pergunta;
2. a pergunta atual é anexada ao prompt mesmo se a estratégia não a devolver;
3. o modelo gera a fala do assistente;
4. o par (usuário, assistente) é gravado na memória.

Controle de execução dos testes: `FakeLLM` e `FakeEmbedder`. O embedder falso é hashing de tokens, não um modelo semântico. Ele só torna o recall lexical reproduzível sem rede. Provedor real não está implementado; pedir outro valor em `LLM_PROVIDER` ou `EMBEDDING_PROVIDER` gera erro.

Fora deste PR: dataset com gabarito de resposta, cálculo de métricas, chamada de API e qualquer tabela de resultado.

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Os defaults do código já são offline (`LLM_PROVIDER=fake`, `EMBEDDING_PROVIDER=fake`). O processo não lê `.env` sozinho. Para sobrescrever um tamanho, exporte a variável ou carregue o modelo:

```bash
cp .env.example .env
set -a && source .env && set +a
```

`.env.example` deixa `OPENAI_API_KEY` vazio. Não commite um `.env` com chave.

A integração contínua (GitHub Actions) roda o lint, a checagem de formato e o pytest em Python 3.12.

## O que há no código

- `src/agent_memory_study/memory/` — `WindowMemory`, `SummaryMemory`, `VectorMemory`
- `src/agent_memory_study/llm.py` — interface do modelo e `FakeLLM`
- `src/agent_memory_study/embeddings.py` — interface de embeddings e `FakeEmbedder`
- `src/agent_memory_study/agent.py` — um turno
- `src/agent_memory_study/factory.py` — escolhe a estratégia pelo nome
- `data/dialogues.json` — três conversas sintéticas de exemplo, não o dataset do estudo

O sumarizador plugado por padrão também é falso: concatena o texto e registra a chamada. Serve para o teste do hook de atualização. Não produz o resumo que o experimento vai avaliar.

## Roadmap

- PR2 — dataset de diálogos sintéticos com gabarito (fatos, turno em que foram plantados, sondagens e resposta esperada).
- PR3 — métricas: recall do fato, correção da resposta, tokens, latência e rótulos de falha (fato esquecido, resumo obsoleto, distrator recuperado).
- PR4 — experimentos com as três estratégias e relatório dos resultados medidos.

## Estado

Nenhum experimento foi executado. Não há tabela de resultados porque não há medição para reportar.
