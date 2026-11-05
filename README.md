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

Unidade experimental: um diálogo do dataset v1 (`data/eval/dialogues.v1.jsonl`). Cada sondagem aponta os ids dos fatos de que a resposta depende e traz o gabarito em `reference_answer`. O arquivo `data/dialogues.json` não entra na medição: são três conversas mínimas para o teste do carregador do PR1, sem gabarito e sem tipo de sondagem.

Condições: `window`, `summary` e `vector`, no mesmo laço de agente. Tamanho da janela, limite do buffer de resumo e `top_k` vêm de variável de ambiente.

O que cada turno faz hoje:

1. a estratégia devolve o contexto que considera relevante para a pergunta;
2. a pergunta atual é anexada ao prompt mesmo se a estratégia não a devolver;
3. o modelo gera a fala do assistente;
4. o par (usuário, assistente) é gravado na memória.

Controle de execução dos testes: `FakeLLM` e `FakeEmbedder`. O embedder falso é hashing de tokens, não um modelo semântico. Ele só torna o recall lexical reproduzível sem rede. Provedor real não está implementado; pedir outro valor em `LLM_PROVIDER` ou `EMBEDDING_PROVIDER` gera erro.

O cálculo das métricas, a chamada de API e qualquer tabela de resultado ficam para os PRs seguintes. O dataset com gabarito já está versionado; ele ainda não foi passado pelas três estratégias.

## Dataset de avaliação

Arquivo: `data/eval/dialogues.v1.jsonl` (versão v1). Uma conversa por linha. A v1 tem 25 diálogos sintéticos e 25 sondagens, cinco de cada tipo abaixo. Nenhum turno foi coletado de uma pessoa. O módulo `eval_dialogues` lê esse arquivo no disco, valida o esquema e não abre rede nem chama modelo.

Campos de cada linha:

- `id` — identificador único do diálogo;
- `turns` — mensagens `{role, content}` (`user` ou `assistant`);
- `planted_facts` — `id`, `text`, `turn`, `kind`. `text` é subcadeia literal de `turns[turn]` e de nenhum outro turno. `kind` é `target` ou `distractor`. Quando o fato corrige um alvo anterior, `supersedes` traz o id desse alvo;
- `probes` — `id`, `question`, `reference_answer`, `relevant_fact_ids`, `type`. `question` é o texto exato de um turno `user`. `reference_answer` é o gabarito e contém o literal de cada fato citado.

A pergunta não repete o literal. Como o literal só existe no turno em que foi plantado, a sondagem não entrega a resposta no próprio enunciado. Turnos de enchimento (recall tardio) e o turno da correção também não repetem o literal antigo. Isso separa "o fato ainda está na memória" de "o fato foi dito de novo agora".

A distância de uma sondagem é o índice do turno da pergunta menos o índice do turno do fato. `IMMEDIATE_MAX_GAP` é 2 e `DELAYED_MIN_GAP` é 8.

Tipos:

- `immediate_recall` — um fato alvo, distância no máximo 2. Isola o caso em que o fato ainda está entre as últimas mensagens.
- `delayed_recall` — um fato alvo, distância no mínimo 8, com enchimento que não contém o literal. Isola o caso em que várias mensagens se interpõem e o valor não foi repetido.
- `fact_update` — o fato citado substitui outro (`supersedes`). O gabarito traz o valor novo e não traz o literal substituído. O turno da correção não repete o valor antigo. Isola o caso em que a resposta correta é o valor novo.
- `multi_hop` — dois fatos alvo em turnos diferentes; o gabarito contém os dois literais. Isola o caso em que a resposta junta duas peças ditas em momentos diferentes.
- `distractor` — um fato alvo e, num turno estritamente entre ele e a pergunta, um fato `distractor` com outro valor. O gabarito é o alvo e não contém o literal do distrator. Isola o caso em que um valor parecido aparece no meio e não deve sobrescrever o fato plantado.

O validador recusa id de diálogo repetido, gabarito vazio, literal ausente do turno indicado, literal repetido noutro turno, sondagem que não vem depois dos fatos que ela cita, tipo fora da lista e rótulo que não obedece à geometria do tipo (lacuna curta demais para `delayed_recall`, `fact_update` sem `supersedes`, `multi_hop` sem dois turnos distintos, `distractor` sem fato distrator entre o alvo e a pergunta).

## Como rodar

Requisitos: Python 3.11+ e [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

O pytest valida `data/eval/dialogues.v1.jsonl` junto com o restante da suíte. Não precisa de chave nem de rede.

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
- `src/agent_memory_study/eval_dialogues.py` — esquema e validação offline do dataset v1
- `data/eval/dialogues.v1.jsonl` — diálogos do estudo
- `data/dialogues.json` — três conversas de exemplo do PR1, não o dataset do estudo

O sumarizador plugado por padrão também é falso: concatena o texto e registra a chamada. Serve para o teste do hook de atualização. Não produz o resumo que o experimento vai avaliar.

## Roadmap

- PR2 — feito. Dataset versionado em `data/eval/dialogues.v1.jsonl`: fatos plantados, sondagens com gabarito e cinco tipos (recall imediato, recall tardio, correção, multi-hop e distrator).
- PR3 — métricas: recall do fato, correção da resposta, tokens, latência e rótulos de falha (fato esquecido, resumo obsoleto, distrator recuperado).
- PR4 — experimentos com as três estratégias e relatório dos resultados medidos.

## Estado

O dataset v1 está no repositório e o validador confere o esquema offline. Nenhuma estratégia foi medida em cima dele. Não há tabela de resultados porque não há medição para reportar.
