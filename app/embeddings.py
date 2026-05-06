from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def carregar_modelo_embeddings(
    nome_modelo: str = DEFAULT_EMBEDDING_MODEL,
) -> SentenceTransformer:
    """
    Carrega e retorna o modelo de embeddings.
    """
    return SentenceTransformer(nome_modelo)


def extrair_textos_chunks(chunks: Sequence[dict]) -> list[str]:
    """
    Extrai apenas os textos dos chunks.
    """
    return [item["chunk"] for item in chunks]


def gerar_embeddings_textos(
    modelo: SentenceTransformer,
    textos: Sequence[str],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Gera embeddings normalizados para uma lista de textos.
    """
    if not textos:
        return np.empty((0, 0), dtype=np.float32)

    embeddings = modelo.encode(
        list(textos),
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype(np.float32)


def gerar_embeddings_chunks(
    modelo: SentenceTransformer,
    chunks: Sequence[dict],
    batch_size: int = 32,
) -> np.ndarray:
    """
    Gera embeddings para os chunks do documento.
    """
    textos_chunks = extrair_textos_chunks(chunks)
    return gerar_embeddings_textos(
        modelo=modelo,
        textos=textos_chunks,
        batch_size=batch_size,
    )


def gerar_embedding_pergunta(
    modelo: SentenceTransformer,
    pergunta: str,
) -> np.ndarray:
    """
    Gera o embedding normalizado da pergunta do usuário.
    """
    embeddings = gerar_embeddings_textos(
        modelo=modelo,
        textos=[pergunta],
        batch_size=1,
    )

    if embeddings.size == 0:
        return np.empty((0,), dtype=np.float32)

    return embeddings[0]


def calcular_similaridades_cosseno(
    embedding_pergunta: np.ndarray,
    embeddings_chunks: np.ndarray,
) -> np.ndarray:
    """
    Calcula similaridade por produto escalar entre vetores já normalizados.
    """
    if embedding_pergunta.size == 0 or embeddings_chunks.size == 0:
        return np.array([], dtype=np.float32)

    return np.dot(embeddings_chunks, embedding_pergunta).astype(np.float32)


def buscar_chunks_semantico(
    pergunta: str,
    chunks: Sequence[dict],
    embeddings_chunks: np.ndarray,
    modelo: SentenceTransformer,
    top_k: int = 5,
) -> list[dict]:
    """
    Retorna os chunks semanticamente mais próximos da pergunta.
    """
    if not pergunta or not pergunta.strip():
        return []

    if not chunks:
        return []

    if embeddings_chunks.size == 0:
        return []

    if len(chunks) != len(embeddings_chunks):
        raise ValueError(
            "A quantidade de chunks e embeddings deve ser a mesma."
        )

    embedding_pergunta = gerar_embedding_pergunta(
        modelo=modelo,
        pergunta=pergunta,
    )

    similaridades = calcular_similaridades_cosseno(
        embedding_pergunta=embedding_pergunta,
        embeddings_chunks=embeddings_chunks,
    )

    if similaridades.size == 0:
        return []

    indices_ordenados = np.argsort(similaridades)[::-1][:top_k]
    resultados = []

    for indice_array in indices_ordenados:
        item = chunks[int(indice_array)]
        score = float(similaridades[int(indice_array)])

        resultados.append(
            {
                "indice": item["indice"],
                "chunk_id_documento": item["chunk_id_documento"],
                "arquivo": item["arquivo"],
                "pagina": item["pagina"],
                "chunk": item["chunk"],
                "inicio_palavra": item["inicio_palavra"],
                "fim_palavra": item["fim_palavra"],
                "score_semantico": round(score, 4),
            }
        )

    return resultados