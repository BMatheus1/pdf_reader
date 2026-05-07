from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from chunking import dividir_em_chunks
from embeddings import (
    carregar_modelo_embeddings,
    gerar_embeddings_chunks,
)
from ingest import extrair_paginas_pdf, juntar_texto_paginas


INDEX_SCHEMA_VERSION = "v1"
INDEX_BASE_DIR = Path(".dev_ai_workspace_index")


def garantir_diretorio_base_indice() -> Path:
    """
    Garante a existência do diretório base dos índices persistentes.
    """
    INDEX_BASE_DIR.mkdir(parents=True, exist_ok=True)
    return INDEX_BASE_DIR


def normalizar_nome_para_caminho(nome: str) -> str:
    """
    Converte um nome de arquivo em um formato seguro para diretório.
    """
    nome_sem_extensao = Path(nome).stem.lower()
    nome_normalizado = re.sub(r"[^a-zA-Z0-9_-]+", "_", nome_sem_extensao).strip("_")
    return nome_normalizado or "documento"


def calcular_hash_arquivo(arquivo_bytes: bytes) -> str:
    """
    Calcula um hash estável para o conteúdo do arquivo.
    """
    return hashlib.sha256(arquivo_bytes).hexdigest()


def calcular_hash_texto(texto: str) -> str:
    """
    Calcula um hash estável para um texto.
    """
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def gerar_id_documento(nome_arquivo: str, arquivo_bytes: bytes) -> str:
    """
    Gera um identificador único para o documento.
    """
    nome_seguro = normalizar_nome_para_caminho(nome_arquivo)
    hash_arquivo = calcular_hash_arquivo(arquivo_bytes)[:12]
    return f"{nome_seguro}__{hash_arquivo}"


def gerar_assinatura_chunking(
    tamanho_chunk: int,
    overlap_chunk: int,
) -> str:
    """
    Gera uma assinatura estável da configuração de chunking.
    """
    return f"chunk_{tamanho_chunk}__overlap_{overlap_chunk}"


def gerar_assinatura_modelo(nome_modelo: str) -> str:
    """
    Gera uma assinatura estável do modelo de embeddings.
    """
    hash_modelo = calcular_hash_texto(nome_modelo)[:12]
    return f"modelo_{hash_modelo}"


def obter_diretorio_documento(documento_id: str) -> Path:
    """
    Retorna o diretório persistente de um documento.
    """
    base_dir = garantir_diretorio_base_indice()
    diretorio = base_dir / "documentos" / documento_id
    diretorio.mkdir(parents=True, exist_ok=True)
    return diretorio


def obter_caminho_documento_processado(
    documento_id: str,
    tamanho_chunk: int,
    overlap_chunk: int,
) -> Path:
    """
    Retorna o caminho do JSON persistente do documento processado.
    """
    assinatura_chunking = gerar_assinatura_chunking(
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
    )
    return obter_diretorio_documento(documento_id) / f"processado__{assinatura_chunking}.json"


def obter_caminho_embeddings_documento(
    documento_id: str,
    tamanho_chunk: int,
    overlap_chunk: int,
    nome_modelo: str,
) -> Path:
    """
    Retorna o caminho do arquivo persistente de embeddings.
    """
    assinatura_chunking = gerar_assinatura_chunking(
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
    )
    assinatura_modelo = gerar_assinatura_modelo(nome_modelo)
    return (
        obter_diretorio_documento(documento_id)
        / f"embeddings__{assinatura_chunking}__{assinatura_modelo}.npy"
    )


def obter_caminho_metadata_embeddings_documento(
    documento_id: str,
    tamanho_chunk: int,
    overlap_chunk: int,
    nome_modelo: str,
) -> Path:
    """
    Retorna o caminho do JSON de metadados dos embeddings.
    """
    assinatura_chunking = gerar_assinatura_chunking(
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
    )
    assinatura_modelo = gerar_assinatura_modelo(nome_modelo)
    return (
        obter_diretorio_documento(documento_id)
        / f"embeddings__{assinatura_chunking}__{assinatura_modelo}.json"
    )


def salvar_json(caminho: Path, dados: dict[str, Any]) -> None:
    """
    Salva um dicionário em JSON.
    """
    caminho.parent.mkdir(parents=True, exist_ok=True)

    with caminho.open("w", encoding="utf-8") as arquivo:
        json.dump(dados, arquivo, ensure_ascii=False, indent=2)


def carregar_json(caminho: Path) -> dict[str, Any]:
    """
    Carrega um JSON do disco.
    """
    with caminho.open("r", encoding="utf-8") as arquivo:
        return json.load(arquivo)


def processar_documento(
    arquivo_bytes: bytes,
    nome_arquivo: str,
    tamanho_chunk: int,
    overlap_chunk: int,
) -> dict[str, Any]:
    """
    Processa um documento PDF em memória.
    """
    paginas = extrair_paginas_pdf(
        arquivo_bytes=arquivo_bytes,
        nome_arquivo=nome_arquivo,
    )
    texto_completo = juntar_texto_paginas(paginas)
    chunks = dividir_em_chunks(
        paginas=paginas,
        tamanho=tamanho_chunk,
        overlap=overlap_chunk,
    )

    return {
        "nome_arquivo": nome_arquivo,
        "paginas": paginas,
        "texto_completo": texto_completo,
        "chunks": chunks,
    }


def montar_documento_persistente(
    documento_id: str,
    arquivo_hash: str,
    nome_arquivo: str,
    tamanho_chunk: int,
    overlap_chunk: int,
    dados_documento: dict[str, Any],
) -> dict[str, Any]:
    """
    Monta a estrutura persistida do documento.
    """
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "documento_id": documento_id,
        "arquivo_hash": arquivo_hash,
        "nome_arquivo": nome_arquivo,
        "tamanho_chunk": tamanho_chunk,
        "overlap_chunk": overlap_chunk,
        "paginas": dados_documento["paginas"],
        "texto_completo": dados_documento["texto_completo"],
        "chunks": dados_documento["chunks"],
    }


def carregar_ou_processar_documento(
    arquivo_bytes: bytes,
    nome_arquivo: str,
    tamanho_chunk: int,
    overlap_chunk: int,
) -> dict[str, Any]:
    """
    Carrega um documento processado do disco ou processa e persiste.
    """
    arquivo_hash = calcular_hash_arquivo(arquivo_bytes)
    documento_id = gerar_id_documento(
        nome_arquivo=nome_arquivo,
        arquivo_bytes=arquivo_bytes,
    )
    caminho_documento = obter_caminho_documento_processado(
        documento_id=documento_id,
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
    )

    if caminho_documento.exists():
        documento = carregar_json(caminho_documento)
        documento["origem_indice_disco"] = True
        return documento

    dados_documento = processar_documento(
        arquivo_bytes=arquivo_bytes,
        nome_arquivo=nome_arquivo,
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
    )

    documento = montar_documento_persistente(
        documento_id=documento_id,
        arquivo_hash=arquivo_hash,
        nome_arquivo=nome_arquivo,
        tamanho_chunk=tamanho_chunk,
        overlap_chunk=overlap_chunk,
        dados_documento=dados_documento,
    )
    salvar_json(caminho_documento, documento)
    documento["origem_indice_disco"] = False
    return documento


def carregar_ou_processar_documentos(
    arquivos_pdf,
    tamanho_chunk: int,
    overlap_chunk: int,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """
    Carrega ou processa todos os documentos enviados.
    """
    documentos: list[dict[str, Any]] = []
    estatisticas = {
        "documentos_carregados_do_disco": 0,
        "documentos_processados_agora": 0,
    }

    for arquivo in arquivos_pdf:
        arquivo_bytes = arquivo.getvalue()

        documento = carregar_ou_processar_documento(
            arquivo_bytes=arquivo_bytes,
            nome_arquivo=arquivo.name,
            tamanho_chunk=tamanho_chunk,
            overlap_chunk=overlap_chunk,
        )

        if documento.get("paginas") and documento.get("texto_completo", "").strip():
            documentos.append(documento)

            if documento.get("origem_indice_disco"):
                estatisticas["documentos_carregados_do_disco"] += 1
            else:
                estatisticas["documentos_processados_agora"] += 1

    return documentos, estatisticas


def carregar_embeddings_do_disco(caminho_embeddings: Path) -> np.ndarray:
    """
    Carrega embeddings persistidos.
    """
    return np.load(caminho_embeddings).astype(np.float32)


def salvar_embeddings_no_disco(
    caminho_embeddings: Path,
    embeddings: np.ndarray,
) -> None:
    """
    Salva embeddings em disco.
    """
    caminho_embeddings.parent.mkdir(parents=True, exist_ok=True)
    np.save(caminho_embeddings, embeddings.astype(np.float32))


def montar_metadata_embeddings(
    documento: dict[str, Any],
    nome_modelo: str,
    embeddings: np.ndarray,
) -> dict[str, Any]:
    """
    Monta os metadados dos embeddings persistidos.
    """
    return {
        "schema_version": INDEX_SCHEMA_VERSION,
        "documento_id": documento["documento_id"],
        "nome_arquivo": documento["nome_arquivo"],
        "nome_modelo": nome_modelo,
        "quantidade_chunks": len(documento["chunks"]),
        "shape_embeddings": list(embeddings.shape),
    }

def metadata_embeddings_valida(
    metadata: dict[str, Any],
    documento: dict[str, Any],
    nome_modelo: str,
    embeddings: np.ndarray,
) -> bool:
    """
    Valida se os metadados persistidos ainda correspondem ao documento atual.
    """
    if metadata.get("schema_version") != INDEX_SCHEMA_VERSION:
        return False

    if metadata.get("documento_id") != documento["documento_id"]:
        return False

    if metadata.get("nome_modelo") != nome_modelo:
        return False

    if metadata.get("quantidade_chunks") != len(documento["chunks"]):
        return False

    if metadata.get("shape_embeddings") != list(embeddings.shape):
        return False

    return True

def carregar_embeddings_persistidos_validos(
    documento: dict[str, Any],
    nome_modelo: str,
) -> np.ndarray | None:
    """
    Carrega embeddings persistidos apenas se os arquivos e metadados estiverem válidos.
    """
    caminho_embeddings = obter_caminho_embeddings_documento(
        documento_id=documento["documento_id"],
        tamanho_chunk=documento["tamanho_chunk"],
        overlap_chunk=documento["overlap_chunk"],
        nome_modelo=nome_modelo,
    )
    caminho_metadata = obter_caminho_metadata_embeddings_documento(
        documento_id=documento["documento_id"],
        tamanho_chunk=documento["tamanho_chunk"],
        overlap_chunk=documento["overlap_chunk"],
        nome_modelo=nome_modelo,
    )

    if not caminho_embeddings.exists() or not caminho_metadata.exists():
        return None

    try:
        embeddings = carregar_embeddings_do_disco(caminho_embeddings)
        metadata = carregar_json(caminho_metadata)
    except Exception:
        return None

    if not metadata_embeddings_valida(
        metadata=metadata,
        documento=documento,
        nome_modelo=nome_modelo,
        embeddings=embeddings,
    ):
        return None

    return embeddings

def carregar_ou_gerar_embeddings_documentos(
    documentos: Sequence[dict[str, Any]],
    nome_modelo: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """
    Carrega embeddings persistidos por documento ou gera quando necessário.
    """
    arrays_embeddings: list[np.ndarray] = []
    estatisticas = {
        "embeddings_carregados_do_disco": 0,
        "embeddings_gerados_agora": 0,
    }

    modelo = None

    for documento in documentos:
        if not documento.get("chunks"):
            continue

        embeddings = carregar_embeddings_persistidos_validos(
            documento=documento,
            nome_modelo=nome_modelo,
        )

        if embeddings is not None:
            arrays_embeddings.append(embeddings)
            estatisticas["embeddings_carregados_do_disco"] += 1
            continue

        caminho_embeddings = obter_caminho_embeddings_documento(
            documento_id=documento["documento_id"],
            tamanho_chunk=documento["tamanho_chunk"],
            overlap_chunk=documento["overlap_chunk"],
            nome_modelo=nome_modelo,
        )
        caminho_metadata = obter_caminho_metadata_embeddings_documento(
            documento_id=documento["documento_id"],
            tamanho_chunk=documento["tamanho_chunk"],
            overlap_chunk=documento["overlap_chunk"],
            nome_modelo=nome_modelo,
        )

        if modelo is None:
            modelo = carregar_modelo_embeddings(nome_modelo)

        embeddings = gerar_embeddings_chunks(
            modelo=modelo,
            chunks=documento["chunks"],
        )

        salvar_embeddings_no_disco(
            caminho_embeddings=caminho_embeddings,
            embeddings=embeddings,
        )
        salvar_json(
            caminho=caminho_metadata,
            dados=montar_metadata_embeddings(
                documento=documento,
                nome_modelo=nome_modelo,
                embeddings=embeddings,
            ),
        )

        arrays_embeddings.append(embeddings)
        estatisticas["embeddings_gerados_agora"] += 1

    if not arrays_embeddings:
        return np.empty((0, 0), dtype=np.float32), estatisticas

    embeddings_consolidados = np.vstack(arrays_embeddings).astype(np.float32)
    return embeddings_consolidados, estatisticas