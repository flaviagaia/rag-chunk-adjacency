"""Demo: segmentação fixa sem vs com adjacência de chunks (~1s).

    python src/demo.py
"""

from __future__ import annotations

from pathlib import Path

from adjacency import Retriever, adjacency, completude, load_chunks, naive

ROOT = Path(__file__).parent.parent
SIZE = 100  # tamanho do chunk (caracteres); corta o sentido de propósito

# Cada consulta tem "spans": partes que PRECISAM estar no contexto para a resposta
# ficar completa. Em Q1 e Q2 elas caem em chunks vizinhos; em Q3, no mesmo chunk.
CONSULTAS = [
    ("Qual a condicao para o pagamento da segunda parcela?",
     ["segunda parcela", "agosto", "aprovacao da prestacao de contas"]),
    ("Como e feita a adesao ao programa e ate quando?",
     ["adesao", "gestor municipal", "fevereiro"]),
    ("Qual o numero minimo de alunos para a escola ser elegivel?",
     ["elegiveis", "cinquenta alunos"]),
]


def main() -> None:
    chunks = load_chunks(ROOT / "data" / "regulamento.txt", size=SIZE)
    r = Retriever(chunks)

    print("=" * 78)
    print(f"Chunk adjacency: reconstruir o contexto cortado (chunks de ~{SIZE} chars)")
    print("=" * 78)

    soma = {"naive": 0.0, "adj": 0.0}
    for q, spans in CONSULTAS:
        cn = naive(r, q)
        ca = adjacency(r, q, window=1)
        c_naive = completude(cn, spans)
        c_adj = completude(ca, spans)
        soma["naive"] += c_naive
        soma["adj"] += c_adj
        print(f"\nP: {q}")
        print(f"   ingênuo  (completude {c_naive:.0%}): {cn}")
        print(f"   adjacência (completude {c_adj:.0%}): {ca}")

    n = len(CONSULTAS)
    print("\n" + "-" * 78)
    print(f"Completude média do contexto: ingênuo {soma['naive']/n:.0%}, "
          f"adjacência {soma['adj']/n:.0%}")
    print("(adjacência nunca remove o chunk certo; só acrescenta os vizinhos)")


if __name__ == "__main__":
    main()
