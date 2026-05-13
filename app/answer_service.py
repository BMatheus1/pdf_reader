from __future__ import annotations

import re
from typing import Any

from lexical_search import extrair_termos_relevantes, normalizar_texto

ResultadoBusca = dict[str, Any]


RESPOSTA_SEM_RESULTADOS = (
    "Não encontrei evidências suficientes para responder com segurança com base "
    "nos trechos recuperados."
)


def limpar_espacos(texto: str) -> str:
    """
    Remove espaços duplicados e quebras de linha desnecessárias.
    """
    return re.sub(r"\s+", " ", texto or "").strip()


def quebrar_em_frases(texto: str) -> list[str]:
    """
    Divide um texto em frases curtas para facilitar a seleção de evidências.
    """
    texto_limpo = limpar_espacos(texto)

    if not texto_limpo:
        return []

    frases = re.split(r"(?<=[\.\!\?\;\:])\s+", texto_limpo)
    frases = [frase.strip(" -•\t") for frase in frases if frase.strip()]

    if frases:
        return frases

    return [texto_limpo]


def obter_score_base_resultado(resultado: ResultadoBusca) -> float:
    """
    Retorna o score mais relevante disponível no resultado.
    """
    for campo in (
        "score_final_rerankeado",
        "score_hibrido",
        "score_semantico",
        "score",
    ):
        if campo in resultado:
            return float(resultado.get(campo, 0.0))

    return 0.0


def contar_termos_encontrados(
    frase_normalizada: str,
    termos_busca: list[str],
) -> list[str]:
    """
    Retorna os termos da busca encontrados na frase.
    """
    return [
        termo
        for termo in termos_busca
        if re.search(rf"(?<!\w){re.escape(termo)}(?!\w)", frase_normalizada)
    ]


def frase_eh_valida_para_resposta(frase: str) -> bool:
    """
    Remove frases muito curtas ou pouco informativas da síntese final.
    """
    frase_limpa = limpar_espacos(frase)

    if len(frase_limpa) < 25:
        return False

    if len(frase_limpa.split()) < 4:
        return False

    return True


def pontuar_frase(
    frase: str,
    termos_busca: list[str],
    score_base_resultado: float,
) -> float:
    """
    Calcula uma pontuação simples para ranquear frases candidatas à resposta final.
    """
    frase_normalizada = normalizar_texto(frase)

    if not frase_normalizada:
        return 0.0

    termos_encontrados = contar_termos_encontrados(
        frase_normalizada=frase_normalizada,
        termos_busca=termos_busca,
    )

    if not termos_encontrados:
        return 0.0

    cobertura = len(set(termos_encontrados)) / max(len(set(termos_busca)), 1)
    ocorrencias = sum(frase_normalizada.count(termo) for termo in termos_encontrados)
    tamanho_penalidade = min(len(frase_normalizada) / 500, 0.35)

    score = (
        cobertura * 4.0
        + ocorrencias * 1.2
        + score_base_resultado * 2.0
        - tamanho_penalidade
    )

    return round(float(score), 4)


def resumir_trecho_para_resposta(
    texto: str,
    limite_caracteres: int = 260,
) -> str:
    """
    Encurta um trecho sem cortar toda a legibilidade.
    """
    texto_limpo = limpar_espacos(texto)

    if len(texto_limpo) <= limite_caracteres:
        return texto_limpo

    trecho = texto_limpo[:limite_caracteres].rsplit(" ", 1)[0].strip()
    return f"{trecho}…"


def extrair_frases_evidencia(
    pergunta: str,
    resultados: list[ResultadoBusca],
    max_frases: int = 5,
) -> list[dict[str, Any]]:
    """
    Extrai frases candidatas à resposta curta.
    """
    if not resultados:
        return []

    termos_busca = extrair_termos_relevantes(pergunta)
    candidatas: list[dict[str, Any]] = []
    frases_vistas: set[str] = set()

    for posicao_resultado, resultado in enumerate(resultados, start=1):
        score_base = obter_score_base_resultado(resultado)

        for frase in quebrar_em_frases(resultado.get("chunk", "")):
            frase_limpa = limpar_espacos(frase)
            frase_chave = normalizar_texto(frase_limpa)

            if not frase_eh_valida_para_resposta(frase_limpa):
                continue

            if not frase_chave or frase_chave in frases_vistas:
                continue

            score_frase = pontuar_frase(
                frase=frase_limpa,
                termos_busca=termos_busca,
                score_base_resultado=score_base,
            )

            if score_frase <= 0:
                continue

            frases_vistas.add(frase_chave)
            candidatas.append(
                {
                    "texto": resumir_trecho_para_resposta(frase_limpa),
                    "arquivo": resultado["arquivo"],
                    "pagina": resultado["pagina"],
                    "score_frase": score_frase,
                    "posicao_resultado": posicao_resultado,
                }
            )

    candidatas.sort(
        key=lambda item: (item["score_frase"], -item["posicao_resultado"]),
        reverse=True,
    )

    return candidatas[:max_frases]


def selecionar_frases_para_resposta(
    frases_evidencia: list[dict[str, Any]],
    max_frases: int = 2,
) -> list[dict[str, Any]]:
    """
    Seleciona frases para a resposta curta de forma conservadora.
    A síntese usa a melhor frase e, se possível, complementa com outra do mesmo arquivo.
    """
    if not frases_evidencia:
        return []

    principal = frases_evidencia[0]
    selecionadas = [principal]

    for candidata in frases_evidencia[1:]:
        if candidata["arquivo"] != principal["arquivo"]:
            continue

        if candidata["texto"] == principal["texto"]:
            continue

        selecionadas.append(candidata)

        if len(selecionadas) >= max_frases:
            break

    return selecionadas


def montar_resposta_curta(
    pergunta: str,
    resultados: list[ResultadoBusca],
    max_frases: int = 2,
) -> str:
    """
    Monta uma resposta curta em linguagem natural com base nas evidências mais fortes.
    A lógica é conservadora para evitar juntar trechos de contextos diferentes.
    """
    if not resultados:
        return RESPOSTA_SEM_RESULTADOS

    frases_evidencia = extrair_frases_evidencia(
        pergunta=pergunta,
        resultados=resultados,
        max_frases=max(max_frases, 4),
    )
    frases_resposta = selecionar_frases_para_resposta(
        frases_evidencia=frases_evidencia,
        max_frases=max_frases,
    )

    if frases_resposta:
        trechos = [item["texto"] for item in frases_resposta]

        if len(trechos) == 1:
            return f"O trecho mais forte encontrado indica que {trechos[0]}"

        resposta = " ".join(trechos)
        return f"Os trechos mais fortes do mesmo documento indicam que {resposta}"

    primeiro_resultado = resultados[0]
    trecho_fallback = resumir_trecho_para_resposta(
        primeiro_resultado.get("chunk", "")
    )

    if trecho_fallback:
        return (
            "Encontrei um trecho relacionado, mas não foi possível gerar uma síntese "
            f"confiável. O melhor trecho encontrado foi: {trecho_fallback}"
        )

    return RESPOSTA_SEM_RESULTADOS


def consolidar_fontes(
    resultados: list[ResultadoBusca],
    max_fontes: int = 5,
) -> list[dict[str, Any]]:
    """
    Agrupa as principais fontes por arquivo e páginas citadas.
    """
    fontes_por_arquivo: dict[str, dict[str, Any]] = {}

    for resultado in resultados:
        arquivo = resultado["arquivo"]
        pagina = int(resultado["pagina"])

        if arquivo not in fontes_por_arquivo:
            fontes_por_arquivo[arquivo] = {
                "arquivo": arquivo,
                "paginas": [],
                "quantidade_trechos": 0,
            }

        if pagina not in fontes_por_arquivo[arquivo]["paginas"]:
            fontes_por_arquivo[arquivo]["paginas"].append(pagina)

        fontes_por_arquivo[arquivo]["quantidade_trechos"] += 1

    fontes = list(fontes_por_arquivo.values())
    fontes.sort(
        key=lambda item: (-item["quantidade_trechos"], min(item["paginas"])),
    )

    for fonte in fontes:
        fonte["paginas"].sort()

    return fontes[:max_fontes]


def selecionar_trechos_apoio(
    resultados: list[ResultadoBusca],
    max_trechos: int = 3,
) -> list[ResultadoBusca]:
    """
    Seleciona os trechos que serão exibidos como apoio da resposta.
    """
    return resultados[:max_trechos]


def gerar_resposta_com_evidencias(
    pergunta: str,
    resultados: list[ResultadoBusca],
    max_frases_resposta: int = 2,
    max_fontes: int = 5,
    max_trechos_apoio: int = 3,
) -> dict[str, Any]:
    """
    Gera uma estrutura pronta para a UI renderizar resposta, fontes e trechos de apoio.
    """
    frases_evidencia = extrair_frases_evidencia(
        pergunta=pergunta,
        resultados=resultados,
        max_frases=max(max_frases_resposta, 4),
    )

    return {
        "resposta_curta": montar_resposta_curta(
            pergunta=pergunta,
            resultados=resultados,
            max_frases=max_frases_resposta,
        ),
        "fontes": consolidar_fontes(
            resultados=resultados,
            max_fontes=max_fontes,
        ),
        "trechos_apoio": selecionar_trechos_apoio(
            resultados=resultados,
            max_trechos=max_trechos_apoio,
        ),
        "frases_evidencia": frases_evidencia[:max_frases_resposta],
    }