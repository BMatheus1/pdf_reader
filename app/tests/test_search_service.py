from search_service import (
    calcular_score_hibrido,
    gerar_chave_unica_chunk,
    normalizar_scores_min_max,
)


def test_gerar_chave_unica_chunk():
    chunk = {
        "arquivo": "doc.pdf",
        "pagina": 2,
        "inicio_palavra": 10,
        "fim_palavra": 30,
    }

    chave = gerar_chave_unica_chunk(chunk)

    assert chave == ("doc.pdf", 2, 10, 30)


def test_normalizar_scores_min_max():
    resultados = [
        {"score": 10},
        {"score": 20},
        {"score": 30},
    ]

    normalizados = normalizar_scores_min_max(
        resultados=resultados,
        campo_score_origem="score",
        campo_score_normalizado="score_norm",
    )

    assert normalizados[0]["score_norm"] == 0.0
    assert normalizados[1]["score_norm"] == 0.5
    assert normalizados[2]["score_norm"] == 1.0


def test_calcular_score_hibrido_com_bonus():
    score = calcular_score_hibrido(
        score_lexical_normalizado=0.8,
        score_semantico_normalizado=0.6,
        peso_lexical=0.7,
        peso_semantico=0.3,
        bonus_presenca_dupla=0.08,
        apareceu_no_lexical=True,
        apareceu_no_semantico=True,
    )

    assert score == 0.82