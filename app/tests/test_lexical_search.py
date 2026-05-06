from lexical_search import (
    normalizar_texto,
    extrair_termos_relevantes,
    buscar_chunks_lexical,
)


def test_normalizar_texto_remove_acentos_e_pontuacao():
    texto = "Sepse, Infecção e Pressão Arterial!"

    resultado = normalizar_texto(texto)

    assert resultado == "sepse infeccao e pressao arterial"


def test_extrair_termos_relevantes_remove_stopwords():
    pergunta = "Qual é a pressão arterial do paciente?"

    termos = extrair_termos_relevantes(pergunta)

    assert "pressao" in termos
    assert "arterial" in termos
    assert "qual" not in termos


def test_buscar_chunks_lexical_retorna_chunk_mais_relevante():
    chunks = [
        {
            "indice": 0,
            "chunk_id_documento": 1,
            "arquivo": "doc.pdf",
            "pagina": 1,
            "chunk": "O paciente apresenta pressão arterial baixa e risco clínico.",
            "inicio_palavra": 0,
            "fim_palavra": 10,
        },
        {
            "indice": 1,
            "chunk_id_documento": 2,
            "arquivo": "doc.pdf",
            "pagina": 2,
            "chunk": "Este trecho fala sobre Docker e README do projeto.",
            "inicio_palavra": 10,
            "fim_palavra": 20,
        },
    ]

    resultados = buscar_chunks_lexical(
        pergunta="pressão arterial baixa",
        chunks=chunks,
        top_k=1,
    )

    assert len(resultados) == 1
    assert resultados[0]["pagina"] == 1
    assert "pressao" in resultados[0]["termos_encontrados"]