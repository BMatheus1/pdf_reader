from __future__ import annotations

import streamlit as st

from app_service import (
    carregar_dados_aplicacao,
    executar_busca_documentos,
    preparar_base_busca,
)
from config import CUSTOM_CSS, DEFAULT_EMBEDDING_MODEL
from index_storage import INDEX_BASE_DIR
from ui_results import (
    renderizar_ajuda_inicial,
    renderizar_cabecalho,
    renderizar_documentos_processados,
    renderizar_resumo_geral,
    renderizar_resumo_indexacao,
    renderizar_resultados_busca,
)
from ui_sidebar import renderizar_sidebar


def configurar_pagina() -> None:
    """
    Configura a página principal do Streamlit.
    """
    st.set_page_config(
        page_title="PDF Reader Inteligente",
        layout="wide",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def renderizar_upload_pdfs():
    """
    Exibe o campo de upload de arquivos PDF.
    """
    return st.file_uploader(
        "Envie um ou mais PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )


def renderizar_filtro_documentos(documentos: list[dict]) -> list[str]:
    """
    Permite ao usuário filtrar em quais documentos a busca será executada.
    """
    nomes_arquivos = [documento["nome_arquivo"] for documento in documentos]

    return st.multiselect(
        "Filtrar busca por arquivos",
        options=nomes_arquivos,
        default=nomes_arquivos,
        help="Deixe todos selecionados para pesquisar em toda a base carregada.",
    )


def renderizar_formulario_busca() -> str:
    """
    Renderiza o formulário de busca principal.
    """
    with st.form("form_busca"):
        pergunta = st.text_input(
            "Pergunta ou termo de busca",
            placeholder="Ex.: o que o documento diz sobre plano de saúde?",
        )
        submitted = st.form_submit_button("Buscar")

    if submitted:
        return pergunta

    return st.session_state.get("ultima_pergunta", "")


def renderizar_informacoes_tecnicas(modo_busca: str) -> None:
    """
    Exibe detalhes técnicos úteis sem poluir a interface principal.
    """
    with st.expander("Detalhes técnicos"):
        st.caption(f"Modo de busca: {modo_busca}")
        if modo_busca in {"Híbrida", "Semântica"}:
            st.caption(f"Modelo de embeddings: {DEFAULT_EMBEDDING_MODEL}")
        st.caption(f"Diretório dos índices: {INDEX_BASE_DIR.resolve()}")


def main() -> None:
    """
    Fluxo principal da aplicação.
    """
    configurar_pagina()
    renderizar_cabecalho()
    renderizar_ajuda_inicial()

    configuracoes = renderizar_sidebar()
    arquivos_pdf = renderizar_upload_pdfs()

    if not arquivos_pdf:
        st.stop()

    try:
        dados = carregar_dados_aplicacao(
            arquivos_pdf=arquivos_pdf,
            tamanho_chunk=int(configuracoes["tamanho_chunk"]),
            overlap_chunk=int(configuracoes["overlap_chunk"]),
            modo_busca=str(configuracoes["modo_busca"]),
            nome_modelo=DEFAULT_EMBEDDING_MODEL,
        )
    except ImportError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Erro ao processar os PDFs: {exc}")
        st.stop()

    documentos = dados["documentos"]
    chunks = dados["chunks"]
    embeddings_chunks = dados["embeddings_chunks"]

    if not documentos:
        st.warning(
            "Nenhum documento com texto legível foi encontrado. "
            "O PDF pode estar vazio, corrompido ou ser apenas imagem."
        )
        st.stop()

    renderizar_resumo_geral(dados["resumo_geral"])
    renderizar_resumo_indexacao(
        dados["estatisticas_documentos"],
        dados["estatisticas_embeddings"],
    )

    arquivos_selecionados = renderizar_filtro_documentos(documentos)
    base_busca = preparar_base_busca(
        chunks=chunks,
        embeddings_chunks=embeddings_chunks,
        arquivos_selecionados=arquivos_selecionados,
        modo_busca=str(configuracoes["modo_busca"]),
    )

    aba_busca, aba_documentos = st.tabs(["🔎 Buscar", "📚 Documentos"])

    with aba_busca:
        pergunta = renderizar_formulario_busca()
        renderizar_informacoes_tecnicas(str(configuracoes["modo_busca"]))

        if pergunta:
            st.session_state["ultima_pergunta"] = pergunta

            try:
                resultados = executar_busca_documentos(
                    pergunta=pergunta,
                    chunks=base_busca["chunks_filtrados"],
                    embeddings_chunks=base_busca["embeddings_filtrados"],
                    top_k=int(configuracoes["top_k"]),
                    modo_busca=str(configuracoes["modo_busca"]),
                    peso_lexical=float(configuracoes["peso_lexical"]),
                    peso_semantico=float(configuracoes["peso_semantico"]),
                    nome_modelo=DEFAULT_EMBEDDING_MODEL,
                )
            except ImportError as exc:
                st.error(str(exc))
                st.stop()
            except Exception as exc:
                st.error(f"Erro ao executar a busca: {exc}")
                st.stop()

            renderizar_resultados_busca(
                resultados=resultados,
                pergunta=pergunta,
                modo_busca=str(configuracoes["modo_busca"]),
            )
        else:
            st.info("Digite uma pergunta para pesquisar nos documentos.")

    with aba_documentos:
        renderizar_documentos_processados(
            documentos=documentos,
            quantidade_chunks_visiveis=int(configuracoes["quantidade_chunks_visiveis"]),
        )


if __name__ == "__main__":
    main()