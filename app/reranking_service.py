from __future__ import annotations

import re
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from config import (
    DEFAULT_INITIAL_SEARCH_WEIGHT,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_WEIGHT,
)
from lexical_search import normalizar_texto

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
else:
    CrossEncoder = Any

ChunkResultado = dict[str, Any]


@lru_cache(maxsize=2)
def carregar_modelo_reranking(
    nome_modelo: str = DEFAULT_RERANKER_MODEL,
) -> CrossEncoder:
    """
    Carrega o modelo de reranking sob demanda e reutiliza em memória.
    """
    try:
        from sentence_transformers import CrossEncoder as _CrossEncoder
    except ImportError as exc:
        raise ImportError(
            "O reranking exige o pacote 'sentence-transformers'. "
            "Instale com: pip install sentence-transformers"
        ) from exc

    return _CrossEncoder(nome_modelo)


def obter_campo_score_inicial(resultados: Sequence[ChunkResultado]) -> str:
    """
    Descobre qual campo de score deve ser usado como base.
    """
    campos_prioritarios = (
        "score_hibrido",
        "score_semantico",
        "score",
    )

    for campo in campos_prioritarios:
        if any(campo in item for item in resultados):
            return campo

    return "score"


def normalizar_lista_scores(scores: Sequence[float]) -> list[float]:
    """
    Normaliza uma lista de scores para o intervalo [0, 1].
    """
    if not scores:
        return []

    score_minimo = min(scores)
    score_maximo = max(scores)

    if score_maximo == score_minimo:
        return [1.0 for _ in scores]

    return [
        round((float(score) - score_minimo) / (score_maximo - score_minimo), 4)
        for score in scores
    ]


def montar_pares_pergunta_chunk(
    pergunta: str,
    resultados: Sequence[ChunkResultado],
) -> list[tuple[str, str]]:
    """
    Cria os pares (pergunta, chunk) usados pelo CrossEncoder.
    """
    return [
        (pergunta, item.get("chunk", ""))
        for item in resultados
    ]


def contem_numero(texto: str) -> bool:
    """
    Verifica se o trecho contém números, datas ou percentuais.
    """
    return bool(re.search(r"\d", texto or ""))


def calcular_bonus_intencao(
    analise_pergunta: dict[str, Any],
    resultado: ChunkResultado,
) -> float:
    """
    Pequenos bônus heurísticos para alinhar o resultado ao tipo de pergunta.
    """
    intencao = analise_pergunta.get("intencao", "geral")
    texto_chunk = resultado.get("chunk", "")
    texto_normalizado = normalizar_texto(texto_chunk)

    if intencao == "localizacao":
        termos = resultado.get("termos_encontrados", [])
        return 0.06 if termos else 0.0

    if intencao == "extracao_objetiva":
        return 0.06 if contem_numero(texto_chunk) else 0.0

    if intencao == "lista":
        sinais_lista = ("•", ";", "\n-", "\n•", ":")
        return 0.05 if any(sinal in texto_chunk for sinal in sinais_lista) else 0.0

    if intencao == "comparacao":
        sinais_comparacao = (
            "diferenca",
            "diferente",
            "compar",
            "enquanto",
            "ja",
            "por outro lado",
        )
        return 0.05 if any(sinal in texto_normalizado for sinal in sinais_comparacao) else 0.0

    if intencao == "resumo":
        quantidade_palavras = len(texto_normalizado.split())
        return min(quantidade_palavras / 4000, 0.04)

    return 0.0


def calcular_score_final_rerankeado(
    score_reranker_normalizado: float,
    score_inicial_normalizado: float,
    peso_reranker: float,
    peso_score_inicial: float,
    bonus_intencao: float,
) -> float:
    """
    Combina o reranker com o score inicial do pipeline.
    """
    score = (
        score_reranker_normalizado * peso_reranker
        + score_inicial_normalizado * peso_score_inicial
        + bonus_intencao
    )
    return round(float(score), 4)


def validar_pesos(
    peso_reranker: float,
    peso_score_inicial: float,
) -> None:
    """
    Garante que os pesos são válidos.
    """
    if peso_reranker < 0 or peso_score_inicial < 0:
        raise ValueError("Os pesos do reranking não podem ser negativos.")

    if np.isclose(peso_reranker + peso_score_inicial, 0.0):
        raise ValueError("A soma dos pesos do reranking deve ser maior que zero.")


def reranquear_resultados(
    pergunta: str,
    resultados: Sequence[ChunkResultado],
    analise_pergunta: dict[str, Any],
    top_k: int,
    nome_modelo: str = DEFAULT_RERANKER_MODEL,
    peso_reranker: float = DEFAULT_RERANKER_WEIGHT,
    peso_score_inicial: float = DEFAULT_INITIAL_SEARCH_WEIGHT,
    batch_size: int = 16,
) -> list[ChunkResultado]:
    """
    Reranqueia os resultados iniciais usando um CrossEncoder.
    """
    if not resultados:
        return []

    validar_pesos(
        peso_reranker=peso_reranker,
        peso_score_inicial=peso_score_inicial,
    )

    campo_score_inicial = obter_campo_score_inicial(resultados)
    modelo = carregar_modelo_reranking(nome_modelo)

    pares = montar_pares_pergunta_chunk(
        pergunta=pergunta,
        resultados=resultados,
    )

    scores_reranker = modelo.predict(
        sentences=pairs,
        batch_size=batch_size,
        show_progress_bar=False,
    )

    scores_reranker_lista = [float(score) for score in np.asarray(scores_reranker).reshape(-1)]
    scores_iniciais = [
        float(item.get(campo_score_inicial, 0.0))
        for item in resultados
    ]

    scores_reranker_normalizados = normalizar_lista_scores(scores_reranker_lista)
    scores_iniciais_normalizados = normalizar_lista_scores(scores_iniciais)

    resultados_atualizados: list[ChunkResultado] = []

    for item, score_reranker, score_reranker_norm, score_inicial_norm in zip(
        resultados,
        scores_reranker_lista,
        scores_reranker_normalizados,
        scores_iniciais_normalizados,
    ):
        bonus_intencao = calcular_bonus_intencao(
            analise_pergunta=analise_pergunta,
            resultado=item,
        )

        item_atualizado = dict(item)
        item_atualizado.update(
            {
                "score_reranker": round(score_reranker, 4),
                "score_reranker_normalizado": round(score_reranker_norm, 4),
                "score_inicial_normalizado": round(score_inicial_norm, 4),
                "score_final_rerankeado": calcular_score_final_rerankeado(
                    score_reranker_normalizado=score_reranker_norm,
                    score_inicial_normalizado=score_inicial_norm,
                    peso_reranker=peso_reranker,
                    peso_score_inicial=peso_score_inicial,
                    bonus_intencao=bonus_intencao,
                ),
                "bonus_intencao": round(bonus_intencao, 4),
                "intencao_pergunta": analise_pergunta.get("intencao", "geral"),
                "descricao_intencao": analise_pergunta.get("descricao_intencao", ""),
                "estilo_resposta": analise_pergunta.get("estilo_resposta", "objetiva"),
            }
        )
        resultados_atualizados.append(item_atualizado)

    resultados_atualizados.sort(
        key=lambda item: (
            item["score_final_rerankeado"],
            item["score_reranker_normalizado"],
            item["score_inicial_normalizado"],
            -item["indice"],
        ),
        reverse=True,
    )

    return resultados_atualizados[:top_k]