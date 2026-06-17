"""Chunk adjacency: recuperar o chunk + os vizinhos para reconstruir o contexto.

A segmentação de tamanho fixo corta o texto sem respeitar o sentido: a condição
de uma regra cai no chunk seguinte ao que casa com a pergunta. O recuperador
ingênuo devolve só o melhor chunk e perde o complemento. A adjacência devolve o
melhor chunk MAIS os vizinhos (anterior/seguinte), remontando a unidade cortada.

- fixed_chunks : segmentação de tamanho fixo (em caracteres, sem quebrar palavra).
- naive        : devolve só o top-1 chunk.
- adjacency    : devolve o top-1 + janela de vizinhos, em ordem do documento.
"""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def fixed_chunks(text: str, size: int) -> list[str]:
    """Segmentação de tamanho fixo (~size caracteres), sem cortar palavra no meio."""
    words = text.split()
    chunks, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > size:
            chunks.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        chunks.append(cur)
    return chunks


def load_chunks(path: Path, size: int = 90) -> list[str]:
    text = " ".join(path.read_text(encoding="utf-8").split())
    return fixed_chunks(text, size)


class Retriever:
    def __init__(self, chunks: list[str]) -> None:
        self.chunks = chunks
        self._vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self._mat = self._vec.fit_transform(chunks)

    def rank(self, query: str) -> list[int]:
        sims = cosine_similarity(self._vec.transform([query]), self._mat).ravel()
        return list(sims.argsort()[::-1])


def naive(retriever: Retriever, query: str) -> str:
    """Só o melhor chunk."""
    return retriever.chunks[retriever.rank(query)[0]]


def adjacency(retriever: Retriever, query: str, window: int = 1) -> str:
    """Melhor chunk + vizinhos (janela), concatenados na ordem do documento."""
    top = retriever.rank(query)[0]
    lo = max(0, top - window)
    hi = min(len(retriever.chunks), top + window + 1)
    return " ".join(retriever.chunks[lo:hi])


def completude(contexto: str, spans: list[str]) -> float:
    """Fração das partes obrigatórias da resposta presentes no contexto."""
    return sum(s in contexto for s in spans) / len(spans)
