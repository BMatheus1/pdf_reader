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


def obter_descricao_modo_busca(modo_busca: str) -> str:
    """
    Retorna uma descrição curta do modo de busca selecionado.
    """
    descricoes = {
        "Lexical": "Melhor para localizar termos exatos, nomes, cláusulas e palavras específicas.",
        "Semântica": "Melhor para encontrar trechos parecidos em significado, mesmo com palavras diferentes.",
        "Híbrida": "Combina correspondência textual com similaridade semântica para um resultado mais equilibrado.",
    }
    return descricoes.get(modo_busca, "")


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

    st.sidebar.caption(obter_descricao_modo_busca(modo_busca))

    tamanho_chunk = st.sidebar.slider(
        "Tamanho do chunk (palavras)",
        min_value=MIN_CHUNK_SIZE,
        max_value=MAX_CHUNK_SIZE,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
        help="Chunks menores tendem a ser mais específicos. Chunks maiores preservam mais contexto.",
    )

    max_overlap = calcular_max_overlap(tamanho_chunk)
    overlap_padrao = min(DEFAULT_CHUNK_OVERLAP, max_overlap)

    overlap_chunk = st.sidebar.slider(
        "Overlap entre chunks (palavras)",
        min_value=0,
        max_value=max_overlap,
        value=overlap_padrao,
        step=10 if max_overlap >= 10 else 1,
        help="Mantém continuidade entre trechos vizinhos para reduzir perda de contexto.",
    )

    top_k = st.sidebar.slider(
        "Quantidade de resultados",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
        step=1,
        help="Quantidade máxima de resultados exibidos na busca.",
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

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "Dica: para perguntas abertas, use Híbrida. "
        "Para localizar expressão exata, use Lexical."
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