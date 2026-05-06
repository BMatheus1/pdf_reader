import re
import unicodedata
from collections import Counter


STOPWORDS_PT = {
    "a", "o", "as", "os", "um", "uma", "uns", "umas",
    "de", "da", "do", "das", "dos",
    "em", "no", "na", "nos", "nas",
    "para", "por", "com", "sem", "sob", "sobre",
    "e", "ou", "mas", "como", "que", "se",
    "ao", "aos", "à", "às",
    "é", "ser", "foi", "são",
    "me", "minha", "meu", "minhas", "meus",
    "qual", "quais", "onde", "quando"
}


def remover_acentos(texto):
    """
    Remove acentos de um texto.
    """
    texto_normalizado = unicodedata.normalize("NFKD", texto)
    return "".join(
        caractere
        for caractere in texto_normalizado
        if not unicodedata.combining(caractere)
    )


def normalizar_texto(texto):
    """
    Padroniza o texto:
    - minúsculas
    - sem acentos
    - sem pontuação
    - sem espaços duplicados
    """
    texto = texto.lower()
    texto = remover_acentos(texto)
    texto = re.sub(r"[^\w\s]", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def tokenizar_texto(texto):
    """
    Quebra o texto normalizado em tokens.
    """
    texto_normalizado = normalizar_texto(texto)

    if not texto_normalizado:
        return []

    return texto_normalizado.split()


def extrair_termos_relevantes(pergunta):
    """
    Remove stopwords simples da pergunta.
    """
    tokens = tokenizar_texto(pergunta)
    termos = [token for token in tokens if token not in STOPWORDS_PT and len(token) > 1]

    if termos:
        return termos

    return tokens


def calcular_bonus_frase_exata(pergunta_normalizada, chunk_normalizado):
    """
    Dá bônus quando a frase completa da pergunta aparece no chunk.
    """
    if not pergunta_normalizada:
        return 0.0

    if len(pergunta_normalizada.split()) < 2:
        return 0.0

    return 4.0 if pergunta_normalizada in chunk_normalizado else 0.0


def calcular_bonus_sequencia_termos(termos_busca, chunk_normalizado):
    """
    Dá bônus quando pares de termos consecutivos aparecem no chunk.
    """
    bonus = 0.0

    for termo_atual, proximo_termo in zip(termos_busca, termos_busca[1:]):
        expressao = f"{termo_atual} {proximo_termo}"
        if expressao in chunk_normalizado:
            bonus += 1.5

    return bonus


def calcular_pontuacao_chunk(pergunta, chunk_texto):
    """
    Calcula a pontuação de relevância de um chunk.
    """
    termos_busca = extrair_termos_relevantes(pergunta)

    if not termos_busca:
        return None

    pergunta_normalizada = normalizar_texto(pergunta)
    chunk_normalizado = normalizar_texto(chunk_texto)
    tokens_chunk = tokenizar_texto(chunk_texto)

    if not tokens_chunk:
        return None

    frequencias = Counter(tokens_chunk)

    termos_encontrados = []
    total_ocorrencias = 0

    for termo in termos_busca:
        ocorrencias = frequencias.get(termo, 0)

        if ocorrencias > 0:
            termos_encontrados.append(termo)
            total_ocorrencias += ocorrencias

    if not termos_encontrados:
        return None

    cobertura = len(termos_encontrados) / len(set(termos_busca))
    densidade = total_ocorrencias / len(tokens_chunk)

    bonus_frase_exata = calcular_bonus_frase_exata(
        pergunta_normalizada=pergunta_normalizada,
        chunk_normalizado=chunk_normalizado,
    )

    bonus_sequencia = calcular_bonus_sequencia_termos(
        termos_busca=termos_busca,
        chunk_normalizado=chunk_normalizado,
    )

    score = (
        cobertura * 10
        + total_ocorrencias * 1.5
        + densidade * 20
        + bonus_frase_exata
        + bonus_sequencia
    )

    return {
        "score": round(score, 2),
        "termos_encontrados": termos_encontrados,
        "cobertura": round(cobertura, 2),
        "ocorrencias": total_ocorrencias,
    }


def buscar_chunks_lexical(pergunta, chunks, top_k=3):
    """
    Retorna os chunks mais relevantes para a pergunta usando busca lexical.
    """
    if not pergunta or not pergunta.strip():
        return []

    if not chunks:
        return []

    resultados = []

    for item in chunks:
        analise = calcular_pontuacao_chunk(
            pergunta=pergunta,
            chunk_texto=item["chunk"],
        )

        if analise is None:
            continue

        resultados.append(
            {
                "indice": item["indice"],
                "chunk_id_documento": item["chunk_id_documento"],
                "arquivo": item["arquivo"],
                "pagina": item["pagina"],
                "chunk": item["chunk"],
                "inicio_palavra": item["inicio_palavra"],
                "fim_palavra": item["fim_palavra"],
                "score": analise["score"],
                "termos_encontrados": analise["termos_encontrados"],
                "cobertura": analise["cobertura"],
                "ocorrencias": analise["ocorrencias"],
            }
        )

    resultados.sort(
        key=lambda item: (
            item["score"],
            item["cobertura"],
            item["ocorrencias"],
            -item["indice"],
        ),
        reverse=True,
    )

    return resultados[:top_k]