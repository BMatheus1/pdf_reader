from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from embeddings import buscar_chunks_semantico
from lexical_search import buscar_chunks_lexical


DEFAULT_PESO_LEXICAL = 0.4
DEFAULT_PESO_SEMANTICO = 0.6
DEFAULT_BONUS_PRESENCA_DUPLA = 0.08
DEFAULT_MULTIPLICADOR_CANDIDATOS = 3

ChunkResultado = dict[str, Any]


def gerar_chave_unica_chunk(item: dict) -> tuple:
    """
    Gera uma chave estável para identificar um chunk.
    """
    return (
        item["arquivo"],
        item["pagina"],
        item["inicio_palavra"],
        item["fim_palavra"],
    )


def normalizar_scores_min_max(
    resultados: Sequence[ChunkResultado],
    campo_score_origem: str,
    campo_score_normalizado: str,
) -> list[ChunkResultado]:
    """
    Normaliza scores no intervalo [0, 1] usando min-max.
    """
    if not resultados:
        return []

    scores = [float(item.get(campo_score_origem, 0.0)) for item in resultados]
    score_minimo = min(scores)
    score_maximo = max(scores)

    resultados_normalizados: list[ChunkResultado] = []

    for item, score in zip(resultados, scores):
        item_atualizado = dict(item)

        if score_maximo == score_minimo:
            item_atualizado[campo_score_normalizado] = 1.0
        else:
            score_normalizado = (score - score_minimo) / (score_maximo - score_minimo)
            item_atualizado[campo_score_normalizado] = round(float(score_normalizado), 4)

        resultados_normalizados.append(item_atualizado)

    return resultados_normalizados


def validar_pesos_busca(
    peso_lexical: float,
    peso_semantico: float,
) -> None:
    """
    Valida os pesos da busca híbrida.
    """
    if peso_lexical < 0 or peso_semantico < 0:
        raise ValueError("Os pesos da busca híbrida não podem ser negativos.")

    soma_pesos = peso_lexical + peso_semantico

    if np.isclose(soma_pesos, 0.0):
        raise ValueError("A soma dos pesos da busca híbrida deve ser maior que zero.")


def criar_mapa_resultados_lexicais(
    resultados_lexicais: Sequence[ChunkResultado],
) -> dict[tuple, ChunkResultado]:
    """
    Indexa os resultados lexicais pela chave única do chunk.
    """
    return {
        gerar_chave_unica_chunk(item): item
        for item in resultados_lexicais
    }


def criar_mapa_resultados_semanticos(
    resultados_semanticos: Sequence[ChunkResultado],
) -> dict[tuple, ChunkResultado]:
    """
    Indexa os resultados semânticos pela chave única do chunk.
    """
    return {
        gerar_chave_unica_chunk(item): item
        for item in resultados_semanticos
    }


def calcular_score_hibrido(
    score_lexical_normalizado: float,
    score_semantico_normalizado: float,
    peso_lexical: float,
    peso_semantico: float,
    bonus_presenca_dupla: float,
    apareceu_no_lexical: bool,
    apareceu_no_semantico: bool,
) -> float:
    """
    Calcula o score final da busca híbrida.
    """
    soma_pesos = peso_lexical + peso_semantico

    score_base = (
        (score_lexical_normalizado * peso_lexical)
        + (score_semantico_normalizado * peso_semantico)
    ) / soma_pesos

    if apareceu_no_lexical and apareceu_no_semantico:
        score_base += bonus_presenca_dupla

    return round(float(score_base), 4)


def combinar_resultados_hibridos(
    resultados_lexicais: Sequence[ChunkResultado],
    resultados_semanticos: Sequence[ChunkResultado],
    peso_lexical: float = DEFAULT_PESO_LEXICAL,
    peso_semantico: float = DEFAULT_PESO_SEMANTICO,
    bonus_presenca_dupla: float = DEFAULT_BONUS_PRESENCA_DUPLA,
) -> list[ChunkResultado]:
    """
    Une resultados lexicais e semânticos em uma única lista reranqueada.
    """
    validar_pesos_busca(
        peso_lexical=peso_lexical,
        peso_semantico=peso_semantico,
    )

    lexicais_normalizados = normalizar_scores_min_max(
        resultados=resultados_lexicais,
        campo_score_origem="score",
        campo_score_normalizado="score_lexical_normalizado",
    )

    semanticos_normalizados = normalizar_scores_min_max(
        resultados=resultados_semanticos,
        campo_score_origem="score_semantico",
        campo_score_normalizado="score_semantico_normalizado",
    )

    mapa_lexical = criar_mapa_resultados_lexicais(lexicais_normalizados)
    mapa_semantico = criar_mapa_resultados_semanticos(semanticos_normalizados)

    chaves_unicas = set(mapa_lexical) | set(mapa_semantico)
    resultados_hibridos: list[ChunkResultado] = []

    for chave in chaves_unicas:
        item_lexical = mapa_lexical.get(chave)
        item_semantico = mapa_semantico.get(chave)

        item_base = dict(item_lexical or item_semantico or {})
        apareceu_no_lexical = item_lexical is not None
        apareceu_no_semantico = item_semantico is not None

        score_lexical = float(item_lexical.get("score", 0.0)) if item_lexical else 0.0
        score_semantico = (
            float(item_semantico.get("score_semantico", 0.0))
            if item_semantico
            else 0.0
        )
        score_lexical_normalizado = (
            float(item_lexical.get("score_lexical_normalizado", 0.0))
            if item_lexical
            else 0.0
        )
        score_semantico_normalizado = (
            float(item_semantico.get("score_semantico_normalizado", 0.0))
            if item_semantico
            else 0.0
        )

        score_hibrido = calcular_score_hibrido(
            score_lexical_normalizado=score_lexical_normalizado,
            score_semantico_normalizado=score_semantico_normalizado,
            peso_lexical=peso_lexical,
            peso_semantico=peso_semantico,
            bonus_presenca_dupla=bonus_presenca_dupla,
            apareceu_no_lexical=apareceu_no_lexical,
            apareceu_no_semantico=apareceu_no_semantico,
        )

        termos_encontrados = []
        if item_lexical:
            termos_encontrados = list(item_lexical.get("termos_encontrados", []))

        item_base.update(
            {
                "score_hibrido": score_hibrido,
                "score_lexical": round(score_lexical, 4),
                "score_semantico": round(score_semantico, 4),
                "score_lexical_normalizado": round(score_lexical_normalizado, 4),
                "score_semantico_normalizado": round(score_semantico_normalizado, 4),
                "termos_encontrados": termos_encontrados,
                "origens_ranking": {
                    "lexical": apareceu_no_lexical,
                    "semantico": apareceu_no_semantico,
                },
            }
        )

        resultados_hibridos.append(item_base)

    resultados_hibridos.sort(
        key=lambda item: (
            item["score_hibrido"],
            item["score_semantico_normalizado"],
            item["score_lexical_normalizado"],
            -item["indice"],
        ),
        reverse=True,
    )

    return resultados_hibridos


def calcular_top_k_candidatos(
    top_k: int,
    multiplicador_candidatos: int = DEFAULT_MULTIPLICADOR_CANDIDATOS,
) -> int:
    """
    Define quantos candidatos cada busca deve retornar antes do reranking.
    """
    return max(top_k, top_k * multiplicador_candidatos)


def buscar_chunks_hibrido(
    pergunta: str,
    chunks: Sequence[dict],
    embeddings_chunks: np.ndarray,
    modelo: SentenceTransformer,
    top_k: int = 5,
    peso_lexical: float = DEFAULT_PESO_LEXICAL,
    peso_semantico: float = DEFAULT_PESO_SEMANTICO,
    bonus_presenca_dupla: float = DEFAULT_BONUS_PRESENCA_DUPLA,
    multiplicador_candidatos: int = DEFAULT_MULTIPLICADOR_CANDIDATOS,
) -> list[ChunkResultado]:
    """
    Executa busca híbrida: lexical + semântica + reranking final.
    """
    if not pergunta or not pergunta.strip():
        return []

    if not chunks:
        return []

    top_k_candidatos = calcular_top_k_candidatos(
        top_k=top_k,
        multiplicador_candidatos=multiplicador_candidatos,
    )

    resultados_lexicais = buscar_chunks_lexical(
        pergunta=pergunta,
        chunks=chunks,
        top_k=top_k_candidatos,
    )

    resultados_semanticos = buscar_chunks_semantico(
        pergunta=pergunta,
        chunks=chunks,
        embeddings_chunks=embeddings_chunks,
        modelo=modelo,
        top_k=top_k_candidatos,
    )

    resultados_hibridos = combinar_resultados_hibridos(
        resultados_lexicais=resultados_lexicais,
        resultados_semanticos=resultados_semanticos,
        peso_lexical=peso_lexical,
        peso_semantico=peso_semantico,
        bonus_presenca_dupla=bonus_presenca_dupla,
    )

    return resultados_hibridos[:top_k]