from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
)
from embeddings import (
    buscar_chunks_semantico,
    carregar_modelo_embeddings,
)
from index_storage import (
    carregar_ou_gerar_embeddings_documentos,
    carregar_ou_processar_documentos,
)
from lexical_search import buscar_chunks_lexical
from query_understanding import analisar_pergunta
from reranking_service import reranquear_resultados
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


def deve_carregar_embeddings(modo_busca: str) -> bool:
    """
    Define se a busca atual depende de embeddings.
    """
    return modo_busca in {"Híbrida", "Semântica"}


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

    embeddings_chunks: np.ndarray | None = None
    estatisticas_embeddings = {
        "embeddings_carregados_do_disco": 0,
        "embeddings_gerados_agora": 0,
    }

    if deve_carregar_embeddings(modo_busca):
        embeddings_chunks, estatisticas_embeddings = (
            carregar_ou_gerar_embeddings_documentos(
                documentos=documentos,
                nome_modelo=nome_modelo,
            )
        )

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
    Se nenhum arquivo for informado, devolve todos.
    """
    if not arquivos_selecionados:
        return chunks

    arquivos_set = set(arquivos_selecionados)
    return [chunk for chunk in chunks if chunk["arquivo"] in arquivos_set]


def criar_array_embeddings_vazio(
    embeddings_chunks: np.ndarray | None,
) -> np.ndarray:
    """
    Cria um array vazio compatível com a dimensão dos embeddings existentes.
    """
    if embeddings_chunks is None or embeddings_chunks.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)

    return np.empty((0, embeddings_chunks.shape[1]), dtype=np.float32)


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

    if not arquivos_selecionados:
        return chunks, embeddings_chunks

    arquivos_set = set(arquivos_selecionados)
    pares_filtrados = [
        (chunk, embedding)
        for chunk, embedding in zip(chunks, embeddings_chunks)
        if chunk["arquivo"] in arquivos_set
    ]

    if not pares_filtrados:
        return [], criar_array_embeddings_vazio(embeddings_chunks)

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
    Prepara a base que será usada na busca.
    """
    if deve_carregar_embeddings(modo_busca):
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


def modo_busca_exige_reranking(modo_busca: str) -> bool:
    """
    Define se o modo de busca deve passar por reranking.
    """
    return modo_busca in {"Semântica", "Híbrida"}


def calcular_top_k_recuperacao_inicial(
    modo_busca: str,
    top_k_final: int,
    multiplicador_candidatos: int,
) -> int:
    """
    Define quantos candidatos devem ser recuperados antes da etapa final.
    No modo lexical, devolvemos apenas o top_k final porque não há reranking.
    """
    if not modo_busca_exige_reranking(modo_busca):
        return top_k_final

    return max(top_k_final, top_k_final * multiplicador_candidatos)


def anexar_metadados_consulta(
    resultados: list[dict[str, Any]],
    analise_pergunta: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    Anexa aos resultados os metadados derivados da pergunta.
    Isso mantém a UI consistente mesmo quando não existe reranking.
    """
    resultados_enriquecidos: list[dict[str, Any]] = []

    for resultado in resultados:
        item = dict(resultado)
        item.update(
            {
                "intencao_pergunta": analise_pergunta.get("intencao", "geral"),
                "descricao_intencao": analise_pergunta.get("descricao_intencao", ""),
                "estilo_resposta": analise_pergunta.get("estilo_resposta", "objetiva"),
            }
        )
        resultados_enriquecidos.append(item)

    return resultados_enriquecidos


def executar_busca_lexical(
    analise_pergunta: dict[str, Any],
    chunks: list[dict[str, Any]],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Executa a busca lexical pura, sem reranking.
    """
    resultados = buscar_chunks_lexical(
        pergunta=analise_pergunta["consulta_lexical"],
        chunks=chunks,
        top_k=top_k,
    )
    return anexar_metadados_consulta(
        resultados=resultados,
        analise_pergunta=analise_pergunta,
    )


def executar_busca_semantica(
    analise_pergunta: dict[str, Any],
    chunks: list[dict[str, Any]],
    embeddings_chunks: np.ndarray,
    top_k_inicial: int,
    nome_modelo: str,
) -> list[dict[str, Any]]:
    """
    Executa a recuperação inicial semântica.
    """
    modelo = carregar_modelo_semantico_cached(nome_modelo)
    return buscar_chunks_semantico(
        pergunta=analise_pergunta["consulta_semantica"],
        chunks=chunks,
        embeddings_chunks=embeddings_chunks,
        modelo=modelo,
        top_k=top_k_inicial,
    )


def executar_busca_hibrida(
    analise_pergunta: dict[str, Any],
    chunks: list[dict[str, Any]],
    embeddings_chunks: np.ndarray,
    top_k: int,
    peso_lexical: float,
    peso_semantico: float,
    nome_modelo: str,
) -> list[dict[str, Any]]:
    """
    Executa a recuperação inicial híbrida.
    """
    modelo = carregar_modelo_semantico_cached(nome_modelo)
    return buscar_chunks_hibrido(
        pergunta=analise_pergunta["pergunta_original"],
        pergunta_lexical=analise_pergunta["consulta_lexical"],
        pergunta_semantica=analise_pergunta["consulta_semantica"],
        chunks=chunks,
        embeddings_chunks=embeddings_chunks,
        modelo=modelo,
        top_k=top_k,
        peso_lexical=peso_lexical,
        peso_semantico=peso_semantico,
        multiplicador_candidatos=int(analise_pergunta["multiplicador_candidatos"]),
    )


def aplicar_etapa_final_busca(
    modo_busca: str,
    resultados_iniciais: list[dict[str, Any]],
    analise_pergunta: dict[str, Any],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Aplica a etapa final da busca.
    - Lexical: apenas enriquece os resultados com metadados da consulta.
    - Semântica/Híbrida: aplica reranking.
    """
    if not resultados_iniciais:
        return []

    if not modo_busca_exige_reranking(modo_busca):
        return anexar_metadados_consulta(
            resultados=resultados_iniciais[:top_k],
            analise_pergunta=analise_pergunta,
        )

    return reranquear_resultados(
        pergunta=analise_pergunta["pergunta_original"],
        resultados=resultados_iniciais,
        analise_pergunta=analise_pergunta,
        top_k=top_k,
        nome_modelo=DEFAULT_RERANKER_MODEL,
    )


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
    Executa a busca completa.

    Fluxo:
    1. entende a pergunta
    2. recupera candidatos conforme o modo de busca
    3. aplica a etapa final (com ou sem reranking)
    """
    if not pergunta or not pergunta.strip():
        return []

    if not chunks:
        return []

    analise_pergunta = analisar_pergunta(pergunta)
    top_k_inicial = calcular_top_k_recuperacao_inicial(
        modo_busca=modo_busca,
        top_k_final=top_k,
        multiplicador_candidatos=int(analise_pergunta["multiplicador_candidatos"]),
    )

    if modo_busca == "Lexical":
        resultados_iniciais = executar_busca_lexical(
            analise_pergunta=analise_pergunta,
            chunks=chunks,
            top_k=top_k_inicial,
        )
        return aplicar_etapa_final_busca(
            modo_busca=modo_busca,
            resultados_iniciais=resultados_iniciais,
            analise_pergunta=analise_pergunta,
            top_k=top_k,
        )

    if embeddings_chunks is None or embeddings_chunks.size == 0:
        return []

    if modo_busca == "Semântica":
        resultados_iniciais = executar_busca_semantica(
            analise_pergunta=analise_pergunta,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            top_k_inicial=top_k_inicial,
            nome_modelo=nome_modelo,
        )
        return aplicar_etapa_final_busca(
            modo_busca=modo_busca,
            resultados_iniciais=resultados_iniciais,
            analise_pergunta=analise_pergunta,
            top_k=top_k,
        )

    if modo_busca == "Híbrida":
        resultados_iniciais = executar_busca_hibrida(
            analise_pergunta=analise_pergunta,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            top_k=top_k,
            peso_lexical=peso_lexical,
            peso_semantico=peso_semantico,
            nome_modelo=nome_modelo,
        )
        return aplicar_etapa_final_busca(
            modo_busca=modo_busca,
            resultados_iniciais=resultados_iniciais,
            analise_pergunta=analise_pergunta,
            top_k=top_k,
        )

    return []