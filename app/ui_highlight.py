from __future__ import annotations

import html
import re
import unicodedata

from lexical_search import extrair_termos_relevantes


def normalizar_texto_com_mapa(texto: str) -> tuple[str, list[int]]:
    """
    Normaliza o texto preservando um mapa entre posições normalizadas e originais.
    Isso permite destacar termos ignorando diferenças de acentuação e caixa.
    """
    texto_normalizado: list[str] = []
    mapa_posicoes: list[int] = []

    for indice_original, caractere in enumerate(texto):
        caractere_normalizado = unicodedata.normalize("NFKD", caractere)
        caractere_sem_acento = "".join(
            parte
            for parte in caractere_normalizado
            if not unicodedata.combining(parte)
        ).lower()

        for parte in caractere_sem_acento:
            texto_normalizado.append(parte)
            mapa_posicoes.append(indice_original)

    return "".join(texto_normalizado), mapa_posicoes


def extrair_spans_termos(texto: str, termos: list[str]) -> list[tuple[int, int]]:
    """
    Localiza os intervalos dos termos no texto original.
    """
    if not texto or not termos:
        return []

    texto_normalizado, mapa_posicoes = normalizar_texto_com_mapa(texto)
    spans: list[tuple[int, int]] = []

    for termo in termos:
        termo_normalizado, _ = normalizar_texto_com_mapa(termo)

        if not termo_normalizado.strip():
            continue

        padrao = re.compile(
            rf"(?<!\w){re.escape(termo_normalizado)}(?!\w)"
        )

        for match in padrao.finditer(texto_normalizado):
            inicio_norm, fim_norm = match.span()
            inicio_original = mapa_posicoes[inicio_norm]
            fim_original = mapa_posicoes[fim_norm - 1] + 1
            spans.append((inicio_original, fim_original))

    return mesclar_spans(spans)


def mesclar_spans(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Une intervalos sobrepostos para evitar marcações duplicadas.
    """
    if not spans:
        return []

    spans_ordenados = sorted(spans, key=lambda item: item[0])
    spans_mesclados = [spans_ordenados[0]]

    for inicio, fim in spans_ordenados[1:]:
        ultimo_inicio, ultimo_fim = spans_mesclados[-1]

        if inicio <= ultimo_fim:
            spans_mesclados[-1] = (ultimo_inicio, max(ultimo_fim, fim))
        else:
            spans_mesclados.append((inicio, fim))

    return spans_mesclados


def destacar_termos_html(texto: str, pergunta: str) -> str:
    """
    Retorna o texto com marcação HTML nos termos relevantes da busca.
    """
    if not texto:
        return ""

    termos = extrair_termos_relevantes(pergunta)
    spans = extrair_spans_termos(texto, termos)

    if not spans:
        return html.escape(texto).replace("\n", "<br>")

    partes_html: list[str] = []
    cursor = 0

    for inicio, fim in spans:
        if cursor < inicio:
            partes_html.append(html.escape(texto[cursor:inicio]))

        trecho = html.escape(texto[inicio:fim])
        partes_html.append(f'<mark class="result-highlight">{trecho}</mark>')
        cursor = fim

    if cursor < len(texto):
        partes_html.append(html.escape(texto[cursor:]))

    return "".join(partes_html).replace("\n", "<br>")