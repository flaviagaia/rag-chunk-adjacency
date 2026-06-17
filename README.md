# rag-chunk-adjacency

Adjacência de chunks: recuperar o trecho **mais os vizinhos** para remontar o
contexto que a segmentação de tamanho fixo cortou.

> **Em uma frase:** a segmentação por tamanho fixo divide o texto sem respeitar o
> sentido, então a condição de uma regra cai no chunk seguinte ao que casa com a
> pergunta; trazer o chunk vencedor junto com os vizinhos reconstrói a unidade.

> *Chunk adjacency: fixed-size chunking splits text without respecting meaning, so a
> rule's condition lands in the chunk after the one that matches the query. Returning
> the top chunk together with its neighbors restores the cut context. It never
> removes the right chunk; it only adds the surrounding ones.*

---

## O problema

Segmentar por tamanho fixo (N caracteres ou tokens) é o default mais comum em RAG, e
é cego ao sentido. Uma regra do tipo "a parcela é paga em agosto, **condicionada à
aprovação da prestação de contas**" pode ser partida bem no meio: o pedaço que casa
com a pergunta ("segunda parcela", "agosto") fica num chunk, e a condição ("aprovação
da prestação de contas") fica no chunk seguinte. O recuperador devolve só o melhor
chunk e a resposta sai **incompleta**, sem o modelo nem perceber que faltou algo.

## Como funciona (o técnico)

Sobre o mesmo retriever TF-IDF (n-gramas 1–2, sem acento):

- `naive` — devolve só o chunk top-1.
- `adjacency` — devolve o chunk top-1 **mais os vizinhos** (janela `window`),
  concatenados na ordem do documento.

```
adjacency(query, window):
    top = argmax_similaridade(query)
    lo, hi = max(0, top - window), min(N, top + window + 1)
    retorna chunks[lo:hi] juntados na ordem original
```

Métrica de **completude**: para cada consulta há um conjunto de partes obrigatórias
(spans) que precisam estar no contexto para a resposta ficar completa; completude é a
fração delas presente no texto recuperado.

Complexidade: a do retriever, mais um fatiamento `O(window)`. O custo extra é
juntar 2·window chunks a mais no contexto. Barato e sem reindexar nada.

## Resultado (determinístico, offline)

Regulamento fictício (Programa Alfa) segmentado em chunks de ~100 caracteres.

| Consulta                                   | Ingênuo | Adjacência |
| ------------------------------------------ | ------- | ---------- |
| Condição da segunda parcela (atravessa)    | 67%     | **100%**   |
| Como e até quando é a adesão (atravessa)   | 33%     | **100%**   |
| Mínimo de alunos elegíveis (1 chunk só)    | 100%    | 100%       |
| **Completude média do contexto**           | **67%** | **100%**   |

Na consulta que já cabia num chunk, a adjacência mantém 100%: ela **nunca remove** o
chunk certo, só acrescenta os vizinhos.

Rode você mesmo:

```bash
pip install -r requirements.txt
python src/demo.py
python -m pytest -q
```

## Como explicar em 30 segundos

"Cortar o texto em pedaços de tamanho fixo separa a regra da sua condição. A busca
acha o pedaço certo, mas a condição ficou no pedaço de baixo, e a resposta sai pela
metade. Eu trago o pedaço vencedor junto com os vizinhos: remonto o trecho cortado.
Custa quase nada e nunca piora."

## Adjacência vs hierarquia

Não confundir com a árvore normativa (`graphrag-hierarquia-normativa`): aquela pega o
**pai** (caput do artigo a partir do inciso); a adjacência pega os **vizinhos
sequenciais** (chunk anterior/seguinte). São sinais estruturais diferentes e
complementares; o `rag-legal-graph-lite` junta os dois.

## Limitações honestas

- Corpus pequeno e fictício, escolhido para o corte ser claro e reproduzível. O ponto
  é o mecanismo, não a magnitude.
- A métrica de completude usa presença de spans (substring), proxy simples de "a
  informação está no contexto". Não mede se o gerador usa bem o contexto.
- Janela maior aumenta a completude mas também o ruído e o custo de contexto; aqui
  `window=1` já basta. Em produção isso é um hiperparâmetro.
- Segmentar por dispositivo (ver `rag-segmentacao`) ataca a mesma dor pela origem;
  adjacência é a correção barata quando você já está preso ao chunk fixo.

## Referências científicas (crédito aos autores)

- **Lewis et al. (2020).** *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks.* NeurIPS.
- **Gao et al. (2024).** *Retrieval-Augmented Generation for Large Language Models: A
  Survey.* arXiv:2312.10997. Discute granularidade de chunk e janelas de contexto.
- **LlamaIndex / LangChain** popularizaram o padrão de *sentence-window* e
  *parent/child retrieval*, do qual a adjacência é a forma mínima.
- Corpus fictício; nenhuma relação com dados reais.

Bibliografia completa do portfólio em `REFERENCIAS.md`.

---

Part of my LinkedIn series on efficient RAG → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
