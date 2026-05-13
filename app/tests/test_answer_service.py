from answer_service import (
    RESPOSTA_SEM_RESULTADOS,
    gerar_resposta_com_evidencias,
    montar_resposta_curta,
)


def test_montar_resposta_curta_sem_resultados():
    resposta = montar_resposta_curta(
        pergunta="qual é o prazo?",
        resultados=[],
    )

    assert resposta == RESPOSTA_SEM_RESULTADOS


def test_montar_resposta_curta_prioriza_mesmo_documento():
    resultados = [
        {
            "arquivo": "contrato_a.pdf",
            "pagina": 1,
            "chunk": "O prazo contratual para pagamento é de 30 dias corridos após a emissão da nota fiscal.",
            "score": 12.0,
        },
        {
            "arquivo": "contrato_b.pdf",
            "pagina": 8,
            "chunk": "A multa por atraso será aplicada em caso de inadimplência.",
            "score": 11.0,
        },
    ]

    resposta = montar_resposta_curta(
        pergunta="qual é o prazo de pagamento?",
        resultados=resultados,
    )

    assert "30 dias" in resposta
    assert "multa" not in resposta.lower()


def test_gerar_resposta_com_evidencias_consolida_fontes_e_trechos():
    resultados = [
        {
            "arquivo": "contrato.pdf",
            "pagina": 2,
            "chunk": "O prazo de pagamento é de 30 dias.",
            "score": 10.0,
        },
        {
            "arquivo": "contrato.pdf",
            "pagina": 3,
            "chunk": "O reajuste anual será aplicado no mês de janeiro.",
            "score": 8.0,
        },
    ]

    resposta = gerar_resposta_com_evidencias(
        pergunta="qual é o prazo de pagamento?",
        resultados=resultados,
    )

    assert resposta["fontes"][0]["arquivo"] == "contrato.pdf"
    assert resposta["fontes"][0]["paginas"] == [2, 3]
    assert len(resposta["trechos_apoio"]) == 2