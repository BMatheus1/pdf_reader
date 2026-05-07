from __future__ import annotations

import streamlit as st

from config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEARCH_MODE,
    DEFAULT_SEMANTIC_WEIGHT,
    DEFAULT_TOP_K,
    DEFAULT_VISIBLE_CHUNKS_PER_DOCUMENT,
    MAX_CHUNK_OVERLAP,
    MAX_CHUNK_SIZE,
    MAX_TOP_K,
    MAX_VISIBLE_CHUNKS_PER_DOCUMENT,
    MIN_CHUNK_SIZE,
    SEARCH_MODES,
)


def calcular_max_overlap(tamanho_chunk: int) -> int:
    """
    Define o máximo de overlap permitido com segurança.
    """
    return max(0, min(MAX_CHUNK_OVERLAP, tamanho_chunk - 1))


def renderizar_sidebar() -> dict[str, float | int | str]:
    """
    Renderiza a barra lateral com as configurações da aplicação.
    """
    st.sidebar.header("⚙️ Configurações")

    modo_busca = st.sidebar.radio(
        "Modo de busca",
        options=SEARCH_MODES,
        index=SEARCH_MODES.index(DEFAULT_SEARCH_MODE),
    )

    tamanho_chunk = st.sidebar.slider(
        "Tamanho do chunk (palavras)",
        min_value=MIN_CHUNK_SIZE,
        max_value=MAX_CHUNK_SIZE,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
    )

    max_overlap = calcular_max_overlap(tamanho_chunk)
    overlap_padrao = min(DEFAULT_CHUNK_OVERLAP, max_overlap)

    overlap_chunk = st.sidebar.slider(
        "Overlap entre chunks (palavras)",
        min_value=0,
        max_value=max_overlap,
        value=overlap_padrao,
        step=10 if max_overlap >= 10 else 1,
    )

    top_k = st.sidebar.slider(
        "Quantidade de resultados",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
    )

    quantidade_chunks_visiveis = st.sidebar.slider(
        "Prévia de chunks por documento",
        min_value=1,
        max_value=MAX_VISIBLE_CHUNKS_PER_DOCUMENT,
        value=DEFAULT_VISIBLE_CHUNKS_PER_DOCUMENT,
        step=1,
    )

    peso_semantico = DEFAULT_SEMANTIC_WEIGHT
    peso_lexical = round(1 - peso_semantico, 2)

    if modo_busca == "Híbrida":
        percentual_semantico = st.sidebar.slider(
            "Peso da busca semântica (%)",
            min_value=0,
            max_value=100,
            value=int(DEFAULT_SEMANTIC_WEIGHT * 100),
            step=5,
        )
        peso_semantico = percentual_semantico / 100
        peso_lexical = round(1 - peso_semantico, 2)

        st.sidebar.caption(
            f"Peso lexical: {int(peso_lexical * 100)}% | "
            f"Peso semântico: {int(peso_semantico * 100)}%"
        )

    return {
        "modo_busca": modo_busca,
        "tamanho_chunk": tamanho_chunk,
        "overlap_chunk": overlap_chunk,
        "top_k": top_k,
        "quantidade_chunks_visiveis": quantidade_chunks_visiveis,
        "peso_lexical": peso_lexical,
        "peso_semantico": peso_semantico,
    }