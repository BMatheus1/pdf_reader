import numpy as np

import app_service


def _analise_pergunta_fake(pergunta: str) -> dict:
    return {
        "pergunta_original": pergunta,
        "consulta_lexical": pergunta,
        "consulta_semantica": pergunta,
        "multiplicador_candidatos": 4,
        "intencao": "extracao_objetiva",
        "descricao_intencao": "extrair uma informação específica",
        "estilo_resposta": "objetiva",
    }


def test_calcular_top_k_recuperacao_inicial_respeita_modo_lexical():
    resultado = app_service.calcular_top_k_recuperacao_inicial(
        modo_busca="Lexical",
        top_k_final=3,
        multiplicador_candidatos=5,
    )

    assert resultado == 3


def test_executar_busca_documentos_lexical_nao_chama_reranking(monkeypatch):
    chunks = [
        {
            "indice": 0,
            "chunk_id_documento": 1,
            "arquivo": "contrato.pdf",
            "pagina": 2,
            "chunk": "O prazo de pagamento é de 30 dias.",
            "inicio_palavra": 0,
            "fim_palavra": 7,
        }
    ]

    monkeypatch.setattr(app_service, "analisar_pergunta", _analise_pergunta_fake)
    monkeypatch.setattr(
        app_service,
        "buscar_chunks_lexical",
        lambda pergunta, chunks, top_k: [
            {
                "indice": 0,
                "chunk_id_documento": 1,
                "arquivo": "contrato.pdf",
                "pagina": 2,
                "chunk": "O prazo de pagamento é de 30 dias.",
                "inicio_palavra": 0,
                "fim_palavra": 7,
                "score": 11.5,
                "termos_encontrados": ["prazo", "pagamento"],
                "cobertura": 1.0,
                "ocorrencias": 2,
            }
        ],
    )

    chamadas = {"reranking": 0}

    def fake_reranking(**kwargs):
        chamadas["reranking"] += 1
        return []

    monkeypatch.setattr(app_service, "reranquear_resultados", fake_reranking)

    resultados = app_service.executar_busca_documentos(
        pergunta="qual é o prazo de pagamento?",
        chunks=chunks,
        embeddings_chunks=None,
        top_k=3,
        modo_busca="Lexical",
        peso_lexical=1.0,
        peso_semantico=0.0,
    )

    assert chamadas["reranking"] == 0
    assert len(resultados) == 1
    assert resultados[0]["descricao_intencao"] == "extrair uma informação específica"


def test_executar_busca_documentos_semantica_chama_reranking(monkeypatch):
    chunks = [
        {
            "indice": 0,
            "chunk_id_documento": 1,
            "arquivo": "contrato.pdf",
            "pagina": 2,
            "chunk": "O prazo de pagamento é de 30 dias.",
            "inicio_palavra": 0,
            "fim_palavra": 7,
        }
    ]
    embeddings = np.ones((1, 3), dtype=np.float32)

    monkeypatch.setattr(app_service, "analisar_pergunta", _analise_pergunta_fake)
    monkeypatch.setattr(
        app_service,
        "carregar_modelo_semantico_cached",
        lambda nome_modelo: object(),
    )
    monkeypatch.setattr(
        app_service,
        "buscar_chunks_semantico",
        lambda pergunta, chunks, embeddings_chunks, modelo, top_k: [
            {
                "indice": 0,
                "chunk_id_documento": 1,
                "arquivo": "contrato.pdf",
                "pagina": 2,
                "chunk": "O prazo de pagamento é de 30 dias.",
                "inicio_palavra": 0,
                "fim_palavra": 7,
                "score_semantico": 0.91,
            }
        ],
    )

    chamadas = {"reranking": 0}

    def fake_reranking(**kwargs):
        chamadas["reranking"] += 1
        return [{"arquivo": "contrato.pdf", "pagina": 2, "chunk": "ok"}]

    monkeypatch.setattr(app_service, "reranquear_resultados", fake_reranking)

    resultados = app_service.executar_busca_documentos(
        pergunta="qual é o prazo de pagamento?",
        chunks=chunks,
        embeddings_chunks=embeddings,
        top_k=3,
        modo_busca="Semântica",
        peso_lexical=0.0,
        peso_semantico=1.0,
    )

    assert chamadas["reranking"] == 1
    assert resultados == [{"arquivo": "contrato.pdf", "pagina": 2, "chunk": "ok"}]