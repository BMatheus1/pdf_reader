from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    buscar_chunks_semantico,
    carregar_modelo_embeddings,
)
from index_storage import (
    carregar_ou_gerar_embeddings_documentos,
    carregar_ou_processar_documentos,
)
from lexical_search import buscar_chunks_lexical
from search_service import buscar_chunks_hibrido


def consolidar_chunks(documentos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Junta os chunks de todos os documentos em uma única lista.
    """
    chunks: list[dict[str, Any]] = []

    for documento in documentos:
        chunks.extend(documento["chunks"])

    return chunks


def calcular_total_palavras(documentos: list[dict[str, Any]]) -> int:
    """
    Calcula o total de palavras de todos os documentos.
    """
    return sum(len(documento["texto_completo"].split()) for documento in documentos)


def montar_resumo_geral(
    documentos: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> dict[str, int]:
    """
    Monta um resumo geral dos documentos processados.
    """
    total_paginas = sum(len(documento["paginas"]) for documento in documentos)
    total_palavras = calcular_total_palavras(documentos)

    return {
        "total_documentos": len(documentos),
        "total_paginas": total_paginas,
        "total_palavras": total_palavras,
        "total_chunks": len(chunks),
    }


def carregar_dados_aplicacao(
    arquivos_pdf,
    tamanho_chunk: int,
    overlap_chunk: int,
    modo_busca: str,
    nome_modelo: str = DEFAULT_EMBEDDING_MODEL,
) -> dict[str, Any]:
    """
    Carrega todos os dados necessários para a aplicação.
    """
    documentos, estatisticas_documentos = carregar_ou_processar_documentos(
        arquivos_pdf=arquivos_pdf,
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
    )

    chunks = consolidar_chunks(documentos)
    resumo_geral = montar_resumo_geral(documentos, chunks)

    if modo_busca in {"Híbrida", "Semântica"}:
        embeddings_chunks, estatisticas_embeddings = carregar_ou_gerar_embeddings_documentos(
            documentos=documentos,
            nome_modelo=nome_modelo,
        )
    else:
        embeddings_chunks = None
        estatisticas_embeddings = {
            "embeddings_carregados_do_disco": 0,
            "embeddings_gerados_agora": 0,
        }

    return {
        "documentos": documentos,
        "chunks": chunks,
        "embeddings_chunks": embeddings_chunks,
        "estatisticas_documentos": estatisticas_documentos,
        "estatisticas_embeddings": estatisticas_embeddings,
        "resumo_geral": resumo_geral,
    }


def filtrar_chunks_por_arquivo(
    chunks: list[dict[str, Any]],
    arquivos_selecionados: list[str],
) -> list[dict[str, Any]]:
    """
    Filtra os chunks pelos arquivos selecionados.
    """
    if not arquivos_selecionados:
        return []

    arquivos_set = set(arquivos_selecionados)
    return [chunk for chunk in chunks if chunk["arquivo"] in arquivos_set]


def filtrar_chunks_e_embeddings_por_arquivo(
    chunks: list[dict[str, Any]],
    embeddings_chunks: np.ndarray | None,
    arquivos_selecionados: list[str],
) -> tuple[list[dict[str, Any]], np.ndarray | None]:
    """
    Filtra chunks e embeddings preservando a mesma ordem entre ambos.
    """
    if embeddings_chunks is None:
        return filtrar_chunks_por_arquivo(chunks, arquivos_selecionados), None

    arquivos_set = set(arquivos_selecionados)

    pares_filtrados = [
        (chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings_chunks)
        if chunk["arquivo"] in arquivos_set
    ]

    if not pares_filtrados:
        return [], np.empty((0, 0), dtype=np.float32)

    chunks_filtrados = [item[0] for item in pares_filtrados]
    embeddings_filtrados = np.vstack(
        [item[1] for item in pares_filtrados]
    ).astype(np.float32)

    return chunks_filtrados, embeddings_filtrados


def preparar_base_busca(
    chunks: list[dict[str, Any]],
    embeddings_chunks: np.ndarray | None,
    arquivos_selecionados: list[str],
    modo_busca: str,
) -> dict[str, Any]:
    """
    Prepara a base de dados que será usada pela busca.
    """
    if modo_busca in {"Híbrida", "Semântica"}:
        chunks_filtrados, embeddings_filtrados = filtrar_chunks_e_embeddings_por_arquivo(
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            arquivos_selecionados=arquivos_selecionados,
        )
    else:
        chunks_filtrados = filtrar_chunks_por_arquivo(
            chunks=chunks,
            arquivos_selecionados=arquivos_selecionados,
        )
        embeddings_filtrados = None

    return {
        "chunks_filtrados": chunks_filtrados,
        "embeddings_filtrados": embeddings_filtrados,
    }


@lru_cache(maxsize=4)
def carregar_modelo_semantico_cached(
    nome_modelo: str = DEFAULT_EMBEDDING_MODEL,
):
    """
    Carrega e reutiliza o modelo semântico em memória.
    """
    return carregar_modelo_embeddings(nome_modelo=nome_modelo)


def executar_busca_documentos(
    pergunta: str,
    chunks: list[dict[str, Any]],
    embeddings_chunks: np.ndarray | None,
    top_k: int,
    modo_busca: str,
    peso_lexical: float,
    peso_semantico: float,
    nome_modelo: str = DEFAULT_EMBEDDING_MODEL,
) -> list[dict[str, Any]]:
    """
    Executa a busca conforme o modo escolhido.
    """
    if not pergunta or not pergunta.strip():
        return []

    if not chunks:
        return []

    if modo_busca == "Lexical":
        return buscar_chunks_lexical(
            pergunta=pergunta,
            chunks=chunks,
            top_k=top_k,
        )

    if embeddings_chunks is None or embeddings_chunks.size == 0:
        return []

    modelo = carregar_modelo_semantico_cached(nome_modelo)

    if modo_busca == "Semântica":
        return buscar_chunks_semantico(
            pergunta=pergunta,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            modelo=modelo,
            top_k=top_k,
        )

    if modo_busca == "Híbrida":
        return buscar_chunks_hibrido(
            pergunta=pergunta,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            modelo=modelo,
            top_k=top_k,
            peso_lexical=peso_lexical,
            peso_semantico=peso_semantico,
        )

    return []