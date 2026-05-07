from __future__ import annotations

import html

import streamlit as st

from answer_service import gerar_resposta_com_evidencias
from ui_highlight import destacar_termos_html


def renderizar_cabecalho() -> None:
    """
    Exibe o título e a descrição principal da aplicação.
    """
    st.title("📄 PDF Reader Inteligente")
    st.write(
        "Pesquise em um ou mais PDFs com busca lexical, semântica ou híbrida. "
        "O foco aqui é encontrar trechos úteis com contexto e origem clara."
    )


def renderizar_ajuda_inicial() -> None:
    """
    Exibe instruções iniciais para orientar o usuário.
    """
    st.info(
        "Envie um ou mais PDFs e depois faça perguntas como: "
        "'qual é o prazo?', 'o que o documento diz sobre estabilidade?' "
        "ou 'onde fala de plano de saúde?'."
    )


def renderizar_resumo_geral(resumo_geral: dict[str, int]) -> None:
    """
    Exibe métricas gerais dos documentos enviados.
    """
    st.subheader("📊 Resumo geral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("PDFs enviados", resumo_geral["total_documentos"])

    with col2:
        st.metric("Páginas com texto", resumo_geral["total_paginas"])

    with col3:
        st.metric("Palavras extraídas", resumo_geral["total_palavras"])

    with col4:
        st.metric("Chunks gerados", resumo_geral["total_chunks"])


def renderizar_resumo_indexacao(
    estatisticas_documentos: dict[str, int],
    estatisticas_embeddings: dict[str, int],
) -> None:
    """
    Exibe um resumo da indexação persistente.
    """
    st.subheader("💾 Indexação local")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documentos carregados do disco",
            estatisticas_documentos["documentos_carregados_do_disco"],
        )
        st.metric(
            "Documentos processados agora",
            estatisticas_documentos["documentos_processados_agora"],
        )

    with col2:
        st.metric(
            "Embeddings carregados do disco",
            estatisticas_embeddings["embeddings_carregados_do_disco"],
        )
        st.metric(
            "Embeddings gerados agora",
            estatisticas_embeddings["embeddings_gerados_agora"],
        )

    st.caption(
        "Os índices ficam salvos localmente para evitar reprocessamento nas próximas execuções."
    )


def renderizar_estado_sem_resultados() -> None:
    """
    Exibe uma mensagem amigável quando a busca não retorna trechos relevantes.
    """
    st.warning(
        "Nenhum resultado relevante foi encontrado. "
        "Tente usar termos mais específicos ou reduzir o filtro de arquivos."
    )


def obter_score_principal(resultado: dict, modo_busca: str) -> tuple[str, float]:
    """
    Define qual score principal deve ser exibido para cada modo de busca.
    """
    if "score_final_rerankeado" in resultado:
        return "Score final", float(resultado.get("score_final_rerankeado", 0.0))

    if modo_busca == "Lexical":
        return "Score lexical", float(resultado.get("score", 0.0))

    if modo_busca == "Semântica":
        return "Score semântico", float(resultado.get("score_semantico", 0.0))

    return "Score híbrido", float(resultado.get("score_hibrido", 0.0))

def formatar_paginas(paginas: list[int]) -> str:
    """
    Formata a lista de páginas para exibição amigável.
    """
    return ", ".join(str(pagina) for pagina in paginas)


def renderizar_metadados_resultado(resultado: dict, modo_busca: str) -> None:
    """
    Exibe metadados auxiliares do resultado encontrado.
    """
    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption(f"Arquivo: {resultado['arquivo']}")

    with col2:
        st.caption(f"Página: {resultado['pagina']}")

    with col3:
        st.caption(
            f"Palavras {resultado['inicio_palavra']}–{resultado['fim_palavra']}"
        )

    if modo_busca == "Lexical":
        termos = resultado.get("termos_encontrados", [])
        if termos:
            st.caption("Termos encontrados: " + ", ".join(termos))

    if modo_busca == "Híbrida":
        st.caption(
            "Composição do ranking — "
            f"lexical: {resultado.get('score_lexical', 0.0):.4f} | "
            f"semântico: {resultado.get('score_semantico', 0.0):.4f}"
        )

    if "descricao_intencao" in resultado:
        st.caption(f"Leitura da pergunta: {resultado['descricao_intencao']}")

    if "score_reranker" in resultado:
        st.caption(
            "Reranking — "
            f"cross-encoder: {resultado.get('score_reranker_normalizado', 0.0):.4f} | "
            f"base: {resultado.get('score_inicial_normalizado', 0.0):.4f} | "
            f"bônus intenção: {resultado.get('bonus_intencao', 0.0):.4f}"
        )
        
def renderizar_fontes_utilizadas(fontes: list[dict]) -> None:
    """
    Exibe a lista de fontes usadas na resposta sugerida.
    """
    st.markdown("**Fontes usadas na resposta**")

    for fonte in fontes:
        paginas = formatar_paginas(fonte["paginas"])
        st.markdown(
            f"- `{fonte['arquivo']}` • páginas {paginas} • trechos usados: {fonte['quantidade_trechos']}"
        )


def renderizar_trechos_apoio(trechos_apoio: list[dict], pergunta: str) -> None:
    """
    Exibe os trechos que sustentam a resposta sugerida.
    """
    with st.expander("Ver trechos de apoio", expanded=False):
        for posicao, resultado in enumerate(trechos_apoio, start=1):
            st.markdown(
                f"**{posicao}. {resultado['arquivo']} • pág. {resultado['pagina']}**"
            )
            trecho_html = destacar_termos_html(resultado["chunk"], pergunta)
            st.markdown(
                f'<div class="result-box subtle-box">{trecho_html}</div>',
                unsafe_allow_html=True,
            )


def renderizar_resposta_sugerida(resultados: list[dict], pergunta: str) -> None:
    """
    Exibe uma resposta curta com fontes e trechos de apoio.
    """
    resposta = gerar_resposta_com_evidencias(
        pergunta=pergunta,
        resultados=resultados,
    )

    st.subheader("🧠 Resposta sugerida")
    st.caption(
        "Síntese automática baseada apenas nos trechos recuperados pela busca. "
        "Use as fontes e os trechos de apoio para validar a interpretação."
    )

    resposta_html = html.escape(resposta["resposta_curta"]).replace("\n", "<br>")
    st.markdown(
        f'<div class="result-box subtle-box">{resposta_html}</div>',
        unsafe_allow_html=True,
    )

    renderizar_fontes_utilizadas(resposta["fontes"])
    renderizar_trechos_apoio(
        trechos_apoio=resposta["trechos_apoio"],
        pergunta=pergunta,
    )


def renderizar_resultados_busca(
    resultados: list[dict],
    pergunta: str,
    modo_busca: str,
) -> None:
    """
    Renderiza a resposta sugerida e os resultados detalhados da busca.
    """
    if not resultados:
        st.subheader("🔎 Resultados da busca")
        renderizar_estado_sem_resultados()
        return

    renderizar_resposta_sugerida(
        resultados=resultados,
        pergunta=pergunta,
    )

    st.subheader("🔎 Resultados detalhados")

    for posicao, resultado in enumerate(resultados, start=1):
        nome_score, score_principal = obter_score_principal(resultado, modo_busca)
        titulo = (
            f"{posicao}. {resultado['arquivo']} • pág. {resultado['pagina']} • "
            f"{nome_score}: {score_principal:.4f}"
        )

        with st.expander(titulo, expanded=(posicao == 1)):
            renderizar_metadados_resultado(resultado, modo_busca)
            trecho_html = destacar_termos_html(resultado["chunk"], pergunta)

            st.markdown(
                f'<div class="result-box subtle-box">{trecho_html}</div>',
                unsafe_allow_html=True,
            )


def renderizar_documentos_processados(
    documentos: list[dict],
    quantidade_chunks_visiveis: int,
) -> None:
    """
    Exibe um resumo individual de cada documento e uma prévia dos chunks.
    """
    st.subheader("📚 Documentos processados")

    for documento in documentos:
        with st.expander(documento["nome_arquivo"]):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Páginas", len(documento["paginas"]))

            with col2:
                st.metric("Palavras", len(documento["texto_completo"].split()))

            with col3:
                st.metric("Chunks", len(documento["chunks"]))

            origem = (
                "Índice carregado do disco"
                if documento.get("origem_indice_disco")
                else "Documento processado nesta execução"
            )
            st.caption(origem)

            st.markdown("**Prévia dos chunks**")
            chunks_documento = documento.get("chunks", [])

            if not chunks_documento:
                st.info("Nenhum chunk gerado para este documento.")
                continue

            for chunk in chunks_documento[:quantidade_chunks_visiveis]:
                st.markdown(
                    f"**Página {chunk['pagina']}** • "
                    f"palavras {chunk['inicio_palavra']}–{chunk['fim_palavra']}"
                )
                st.code(chunk["chunk"], language=None)

            with st.expander("Ver texto completo"):
                st.text_area(
                    label=f"Texto completo — {documento['nome_arquivo']}",
                    value=documento["texto_completo"],
                    height=320,
                    key=f"texto_{documento['documento_id']}",
                )