from __future__ import annotations

import re
from typing import Any

from lexical_search import extrair_termos_relevantes, normalizar_texto

AnalisePergunta = dict[str, Any]

PADROES_INTENCAO: dict[str, tuple[str, ...]] = {
    "resumo": (
        r"\bresuma\b",
        r"\bresumo\b",
        r"\bsintetiz",
        r"\bexplique em poucas palavras\b",
        r"\bprincipais pontos\b",
    ),
    "localizacao": (
        r"\bonde\b",
        r"\bem que pagina\b",
        r"\bem qual pagina\b",
        r"\bpagina\b",
        r"\bp[aá]gina\b",
        r"\btrecho\b",
        r"\bcl[aá]usula\b",
        r"\bcap[ií]tulo\b",
        r"\bse[cç][aã]o\b",
    ),
    "comparacao": (
        r"\bcompare\b",
        r"\bcompara\b",
        r"\bdiferen[cç]a\b",
        r"\bdiferen[cç]as\b",
        r"\bversus\b",
        r"\bvs\b",
    ),
    "lista": (
        r"\bliste\b",
        r"\bquais s[aã]o\b",
        r"\bquais sao\b",
        r"\bitens\b",
        r"\bpontos\b",
        r"\brequisitos\b",
        r"\bcondi[cç][oõ]es\b",
    ),
    "extracao_objetiva": (
        r"\bqual\b",
        r"\bquanto\b",
        r"\bquando\b",
        r"\bprazo\b",
        r"\bdata\b",
        r"\bvalor\b",
        r"\bvencimento\b",
        r"\bvig[eê]ncia\b",
        r"\bin[ií]cio\b",
        r"\bfim\b",
        r"\bpercentual\b",
    ),
}

TERMOS_AUXILIARES_POR_INTENCAO: dict[str, tuple[str, ...]] = {
    "resumo": ("resumo", "síntese", "principais pontos"),
    "localizacao": ("página", "trecho", "cláusula", "seção"),
    "comparacao": ("comparação", "diferença", "semelhança"),
    "lista": ("itens", "requisitos", "condições", "pontos"),
    "extracao_objetiva": ("prazo", "data", "valor", "vigência", "percentual"),
    "geral": tuple(),
}

DESCRICAO_INTENCAO: dict[str, str] = {
    "resumo": "resumir o conteúdo mais relevante",
    "localizacao": "localizar onde o assunto aparece no documento",
    "comparacao": "comparar trechos ou condições do documento",
    "lista": "listar itens, requisitos ou condições",
    "extracao_objetiva": "extrair uma informação específica",
    "geral": "buscar contexto geral sobre o assunto",
}

ESTILO_RESPOSTA: dict[str, str] = {
    "resumo": "resumida",
    "localizacao": "localizacao",
    "comparacao": "comparativa",
    "lista": "lista",
    "extracao_objetiva": "objetiva",
    "geral": "objetiva",
}


def deduplicar_preservando_ordem(itens: list[str]) -> list[str]:
    """
    Remove duplicatas preservando a ordem original.
    """
    vistos: set[str] = set()
    resultado: list[str] = []

    for item in itens:
        chave = normalizar_texto(item)

        if not chave or chave in vistos:
            continue

        vistos.add(chave)
        resultado.append(item)

    return resultado


def classificar_intencao(pergunta: str) -> str:
    """
    Classifica a intenção dominante da pergunta.
    """
    pergunta_normalizada = normalizar_texto(pergunta)

    for intencao, padroes in PADROES_INTENCAO.items():
        if any(re.search(padrao, pergunta_normalizada) for padrao in padroes):
            return intencao

    return "geral"


def extrair_termos_numericos(pergunta: str) -> list[str]:
    """
    Extrai datas, números e percentuais que ajudam bastante em perguntas objetivas.
    """
    encontrados = re.findall(r"\b\d+(?:[\/\.-]\d+)*%?\b", pergunta)
    return deduplicar_preservando_ordem(encontrados)


def montar_consulta_lexical(
    pergunta: str,
    intencao: str,
) -> str:
    """
    Monta uma versão enriquecida da pergunta para a busca lexical.
    """
    termos_relevantes = extrair_termos_relevantes(pergunta)
    termos_numericos = extrair_termos_numericos(pergunta)
    termos_auxiliares = list(TERMOS_AUXILIARES_POR_INTENCAO.get(intencao, tuple()))

    termos_expandidos = deduplicar_preservando_ordem(
        termos_relevantes + termos_numericos + termos_auxiliares
    )

    if not termos_expandidos:
        return pergunta.strip()

    return f"{pergunta.strip()} " + " ".join(termos_expandidos)


def sugerir_multiplicador_candidatos(intencao: str) -> int:
    """
    Ajusta o volume de candidatos recuperados antes do reranking.
    """
    if intencao in {"resumo", "comparacao", "lista"}:
        return 5

    if intencao in {"localizacao", "extracao_objetiva"}:
        return 4

    return 3


def analisar_pergunta(pergunta: str) -> AnalisePergunta:
    """
    Gera uma estrutura única com o entendimento da pergunta.
    """
    pergunta_limpa = (pergunta or "").strip()
    pergunta_normalizada = normalizar_texto(pergunta_limpa)
    intencao = classificar_intencao(pergunta_limpa)

    return {
        "pergunta_original": pergunta_limpa,
        "pergunta_normalizada": pergunta_normalizada,
        "intencao": intencao,
        "descricao_intencao": DESCRICAO_INTENCAO[intencao],
        "estilo_resposta": ESTILO_RESPOSTA[intencao],
        "consulta_lexical": montar_consulta_lexical(
            pergunta=pergunta_limpa,
            intencao=intencao,
        ),
        "consulta_semantica": pergunta_limpa,
        "multiplicador_candidatos": sugerir_multiplicador_candidatos(intencao),
        "termos_relevantes": extrair_termos_relevantes(pergunta_limpa),
        "termos_numericos": extrair_termos_numericos(pergunta_limpa),
    }