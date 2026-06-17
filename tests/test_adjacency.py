import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from adjacency import (  # noqa: E402
    Retriever,
    adjacency,
    completude,
    fixed_chunks,
    load_chunks,
    naive,
)

CHUNKS = load_chunks(ROOT / "data" / "regulamento.txt", size=100)
R = Retriever(CHUNKS)

Q_SEGUNDA = "Qual a condicao para o pagamento da segunda parcela?"
SPANS_SEGUNDA = ["segunda parcela", "agosto", "aprovacao da prestacao de contas"]
Q_ELEGIB = "Qual o numero minimo de alunos para a escola ser elegivel?"
SPANS_ELEGIB = ["elegiveis", "cinquenta alunos"]


def test_fixed_chunks_respeita_tamanho_e_palavras():
    ch = fixed_chunks("uma frase curta de teste aqui mesmo", 12)
    assert all(len(c) <= 14 for c in ch)  # ~size, sem cortar palavra
    assert " ".join(ch).split() == "uma frase curta de teste aqui mesmo".split()


def test_naive_perde_o_complemento_da_regra():
    ctx = naive(R, Q_SEGUNDA)
    c = completude(ctx, SPANS_SEGUNDA)
    assert c < 1.0  # o chunk top-1 não tem todos os spans
    assert "aprovacao da prestacao de contas" not in ctx


def test_adjacency_reconstroi_o_contexto():
    ctx = adjacency(R, Q_SEGUNDA, window=1)
    assert completude(ctx, SPANS_SEGUNDA) == 1.0


def test_adjacency_nao_atrapalha_chunk_unico():
    # Resposta cabe num chunk só: naive já acerta e adjacência mantém 100%.
    assert completude(naive(R, Q_ELEGIB), SPANS_ELEGIB) == 1.0
    assert completude(adjacency(R, Q_ELEGIB, window=1), SPANS_ELEGIB) == 1.0


def test_completude_media_melhora():
    consultas = [
        (Q_SEGUNDA, SPANS_SEGUNDA),
        ("Como e feita a adesao ao programa e ate quando?",
         ["adesao", "gestor municipal", "fevereiro"]),
        (Q_ELEGIB, SPANS_ELEGIB),
    ]
    n = len(consultas)
    media_naive = sum(completude(naive(R, q), s) for q, s in consultas) / n
    media_adj = sum(completude(adjacency(R, q, 1), s) for q, s in consultas) / n
    assert media_adj > media_naive
    assert media_adj == 1.0
