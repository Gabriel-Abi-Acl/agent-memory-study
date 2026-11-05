# Estudo de memória em agentes conversacionais

Comparação técnica de três estratégias de memória num agente de diálogo multi-turno: janela deslizante, resumo incremental e recuperação vetorial. Os diálogos são sintéticos. O harness de avaliação está implementado. Este repositório não contém resultado experimental.

## Objetivo

Medir, no mesmo conjunto de diálogos, como cada estratégia se comporta quando uma pergunta tardia depende de um fato dito antes. O harness calcula, por sondagem:

- recall do fato plantado: o contexto devolvido pela estratégia contém o literal;
- correção da resposta contra o gabarito (igualdade normalizada e, se pedido, juiz de paráfrase);
- tokens do prompt, tokens da resposta, latência e custo estimado;
- modo de falha: fato ausente do contexto, resumo que ainda carrega o valor substituído, ou distrator presente no contexto.

Nada disso foi medido com modelo real. Os números de um experimento entram só no PR4. Uma tabela gerada localmente é o traço daquela execução, não o resultado do estudo.

## Hipótese

Ainda não testada. O desenho do dataset e das métricas parte de três expectativas:

1. A janela deslizante conserva o fato enquanto ele está entre as últimas N mensagens e o perde quando a mensagem é expulsa.
2. O resumo gasta menos tokens em diálogos longos e preserva o tema, mas pode descartar um valor literal (nome, código, cor) ou manter um valor antigo depois de uma correção.
3. A recuperação vetorial devolve o enunciado do fato quando a pergunta se aproxima dele no espaço do embedder, e pode devolver um trecho parecido que não contém o fato.

Se alguma dessas expectativas falhar na medição, o relatório deve dizer isso. Elas não são conclusão.

## Metodologia

Unidade experimental: um diálogo do dataset v1 (`data/eval/dialogues.v1.jsonl`). Cada sondagem aponta os ids dos fatos de que a resposta depende e traz o gabarito em `reference_answer`. O arquivo `data/dialogues.json` não entra na medição: são três conversas mínimas para o teste do carregador do PR1, sem gabarito e sem tipo de sondagem.

Condições: `window`, `summary` e `vector`, no mesmo harness. Tamanho da janela, limite do buffer de resumo e `top_k` vêm de variável de ambiente.

O laço de um turno do agente (`Agent.step`) continua o mesmo: a estratégia devolve o contexto, a pergunta atual entra no prompt mesmo se a estratégia não a devolver, o modelo gera a fala e o par é gravado. O runner de avaliação não usa esse laço para reescrever a história. Ele grava os turnos do JSONL como estão e só chama o modelo na pergunta da sondagem. A resposta gerada não substitui o turno seguinte do script. Assim o recall mede o fato plantado, não o texto que o modelo acabou de produzir.

Com `WINDOW_SIZE=8` e a lacuna mínima do recall tardio em 8, o fato plantado no turno 0 ainda cabe na janela quando a pergunta está no turno 8. Para observar expulsão, a janela tem de ser menor que o número de mensagens anteriores à pergunta. No `fact_update` da v1 há quatro mensagens antes da pergunta; com `SUMMARY_RECENT_LIMIT=4` nada é condensado ainda, então o rótulo de resumo obsoleto não acende. Ele acende quando o literal substituído está no texto do resumo, não quando o turno antigo ainda está no buffer literal.

Métricas, por sondagem:

- normalização: NFKC, casefold, pontuação (incluindo hífen) vira espaço, espaços colapsam. Há acerto se o texto normalizado é igual ao gabarito normalizado. Os dois vazios não contam.
- juiz: opcional (`--judge`). Só roda se a normalização falhou. O modelo responde SIM ou NAO. Outra fala conta como erro. Com o provedor fake o juiz não produz paráfrase útil, porque a resposta fake não é SIM nem NAO.
- recall: fração dos fatos em `relevant_fact_ids` cujo `text` aparece, como substring literal, no contexto devolvido antes de indexar a pergunta.
- `forgotten_fact`: algum fato relevante ausente desse contexto.
- `stale_summary`: sondagem `fact_update` cujo literal substituído (`supersedes`) está no miolo da mensagem de sistema do resumo.
- `retrieved_distractor`: sondagem `distractor` em que o literal de um distrator entre o alvo e a pergunta está no contexto. Não exige que o alvo esteja ausente; isso é `forgotten_fact`. Uma sondagem pode receber mais de um rótulo.
- latência: relógio em volta da chamada que gera a resposta da sondagem, já com as tentativas.
- tokens: `usage` da API quando existe; senão teto(caracteres / 4) por mensagem. No `records.jsonl`, a resposta da sondagem fica em `answer_*` e o sumarizador mais os embeddings reais ficam em `memory_*`. O CSV soma os dois em `tokens_prompt` e `tokens_resposta`.
- custo: tokens vezes a tabela de lista em `usage.py` (gpt-4o-mini, gpt-4o, text-embedding-3-small, fake = 0), ou `PROMPT_USD_PER_MILLION` e `COMPLETION_USD_PER_MILLION` para o modelo de chat. Hit de cache custa 0 nesta execução e mantém os tokens registrados. Modelo fora da tabela, sem override, custa 0. Não é fatura.

A agregação escreve uma linha por estratégia e tipo de sondagem presente, mais uma linha `all` por estratégia. Colunas: n, acurácia, recall médio, contagem de cada rótulo, soma de tokens, latência média, soma do custo estimado. Saída em Markdown e CSV. O Markdown abre dizendo que o arquivo é o traço da execução.

Retry: HTTP 408, 429, 500, 502, 503 e 504, e queda de transporte. Espera `Retry-After` em segundos, senão backoff exponencial a partir de `LLM_RETRY_BASE_DELAY_S`, com teto `LLM_RETRY_MAX_DELAY_S`. 400 e 401 não repetem. O cliente não segue redirect e ignora `HTTP(S)_PROXY`, para o Bearer não ir a outro host; um gateway entra por `OPENAI_BASE_URL`. A chave não vai para o repr, para a mensagem de erro nem para o arquivo de cache.

Cache: um JSON por sha256 do modelo, da temperatura 0 e das mensagens, no diretório `--cache-dir` ou `LLM_CACHE_DIR`. Cobre chat de resposta, juiz e resumo. Embedding não entra nesse cache; 429 de embedding usa o mesmo retry. Arquivo corrompido é tratado como miss.

Controle dos testes: `FakeLLM`, `FakeEmbedder` e um juiz scriptado. O embedder falso é hashing de tokens, não um modelo semântico. A suíte não abre rede. Provedor fora de `fake` e `openai` gera erro. `openai` sem `OPENAI_API_KEY` falha na construção do cliente, antes do socket.

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

O pytest valida `data/eval/dialogues.v1.jsonl` e o harness com LLM, embedder e juiz falsos. Não precisa de chave nem de rede. A integração contínua roda o mesmo lint, a checagem de formato e o pytest em Python 3.12.

Os defaults já são offline (`LLM_PROVIDER=fake`, `EMBEDDING_PROVIDER=fake`). O processo não lê `.env` sozinho. Para sobrescrever um tamanho, exporte a variável ou carregue o modelo:

```bash
cp .env.example .env
set -a && source .env && set +a
```

`.env.example` deixa `OPENAI_API_KEY` vazio. Não commite um `.env` com chave.

Avaliação offline, nos dois primeiros diálogos, nas três estratégias. O diretório de saída fica de fora do git. O Markdown e o CSV são o traço dessa execução com o modelo fake, que não responde o gabarito. Não use esses arquivos como resultado do estudo.

```bash
uv run evaluate-memory --strategy all --limit 2 --output-dir eval-out --cache-dir .cache/llm
```

O mesmo comando, no dataset inteiro:

```bash
uv run evaluate-memory --strategy all --output-dir eval-out --cache-dir .cache/llm
```

Arquivos escritos: `records.jsonl` (uma sondagem por linha), `summary.csv` e `summary.md`.

Estratégia única, ou uma lista:

```bash
uv run evaluate-memory --strategy window --strategy vector --output-dir eval-out
uv run python -m agent_memory_study --strategy summary --limit 5 --output-dir eval-out
```

`--fake-reply` só vale para o provedor fake. `--judge` liga o juiz de paráfrase no mesmo cliente. `--dialogues` troca o JSONL; o default é `data/eval/dialogues.v1.jsonl`.

Para uma execução real, mais tarde, com chave só no ambiente:

```bash
export LLM_PROVIDER=openai
export EMBEDDING_PROVIDER=openai
export OPENAI_API_KEY=...   # não grave isto no repositório
export OPENAI_MODEL=gpt-4o-mini
uv run evaluate-memory --strategy all --output-dir eval-out --cache-dir .cache/llm --judge
```

Com `LLM_PROVIDER=openai` o resumo passa a ser pedido ao modelo (`LLMSummarizer`). Com fake, o sumarizador continua concatenando e o teste do hook observa essa chamada. Os dois não são dado de experimento.

## O que há no código

- `src/agent_memory_study/memory/` — `WindowMemory`, `SummaryMemory`, `VectorMemory`, `LLMSummarizer`
- `src/agent_memory_study/llm.py` — interface do modelo e `FakeLLM`
- `src/agent_memory_study/llm_runtime.py` — retry, cache em disco e cliente OpenAI injetável
- `src/agent_memory_study/embeddings.py` — interface de embeddings e `FakeEmbedder`
- `src/agent_memory_study/usage.py` — estimativa de tokens e de custo
- `src/agent_memory_study/metrics.py` — correção, recall, rótulos, agregação, CSV e Markdown
- `src/agent_memory_study/runner.py` — reproduz o diálogo e registra cada sondagem
- `src/agent_memory_study/cli.py` — `evaluate-memory`
- `src/agent_memory_study/agent.py` — um turno
- `src/agent_memory_study/factory.py` — escolhe a estratégia pelo nome
- `src/agent_memory_study/eval_dialogues.py` — esquema e validação offline do dataset v1
- `data/eval/dialogues.v1.jsonl` — diálogos do estudo
- `data/dialogues.json` — três conversas de exemplo do PR1, não o dataset do estudo

## Roadmap

- PR2 — feito. Dataset versionado em `data/eval/dialogues.v1.jsonl`: fatos plantados, sondagens com gabarito e cinco tipos (recall imediato, recall tardio, correção, multi-hop e distrator).
- PR3 — feito. Harness e métricas: recall do fato no estado da memória no momento da sondagem, correção por normalização e juiz opcional, rótulos de falha (fato esquecido, resumo obsoleto, distrator no contexto), tokens, latência e custo estimado, agregação em Markdown e CSV, retry e cache em disco. Sem tabela de resultado do estudo.
- PR4 — experimentos com as três estratégias e relatório dos resultados medidos.

## Estado

O dataset v1 está no repositório, o validador confere o esquema offline e o harness roda as três estratégias em teste, sem rede. Nenhuma estratégia foi medida com modelo real. Não há tabela de resultados do estudo neste repositório.
