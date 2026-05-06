from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from chunking import dividir_em_chunks
from ingest import extrair_paginas_pdf
from lexical_search import buscar_chunks_lexical, normalizar_texto
from search_eval_cases import SEARCH_EVAL_CASES


DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 80
DEFAULT_TOP_K_MAX = 5
DEFAULT_OUTPUT_DIR = Path("search_eval_output")


SearchFn = Callable[[str, int], list[dict[str, Any]]]


def carregar_argumentos() -> argparse.Namespace:
    """
    Lê os argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(
        description="Avalia a qualidade da busca lexical, semântica e híbrida em um PDF."
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Caminho do PDF que será usado na avaliação.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Tamanho do chunk em palavras. Padrão: {DEFAULT_CHUNK_SIZE}.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help=f"Overlap entre chunks em palavras. Padrão: {DEFAULT_OVERLAP}.",
    )
    parser.add_argument(
        "--top-k-max",
        type=int,
        default=DEFAULT_TOP_K_MAX,
        help=f"Quantidade máxima de resultados retornados por pergunta. Padrão: {DEFAULT_TOP_K_MAX}.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Diretório onde os arquivos CSV e JSON serão salvos.",
    )
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Pula avaliação semântica e híbrida.",
    )
    return parser.parse_args()


def carregar_pdf_em_paginas(pdf_path: Path) -> list[dict[str, Any]]:
    """
    Lê o PDF e retorna as páginas extraídas.
    """
    arquivo_bytes = pdf_path.read_bytes()
    return extrair_paginas_pdf(
        arquivo_bytes=arquivo_bytes,
        nome_arquivo=pdf_path.name,
    )


def gerar_chunks_pdf(
    paginas: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
) -> list[dict[str, Any]]:
    """
    Gera os chunks do PDF a partir das páginas extraídas.
    """
    return dividir_em_chunks(
        paginas=paginas,
        tamanho=chunk_size,
        overlap=overlap,
    )


def construir_buscador_lexical(chunks: list[dict[str, Any]]) -> SearchFn:
    """
    Monta o buscador lexical.
    """

    def buscar(pergunta: str, top_k: int) -> list[dict[str, Any]]:
        return buscar_chunks_lexical(
            pergunta=pergunta,
            chunks=chunks,
            top_k=top_k,
        )

    return buscar


def construir_buscadores_semanticos(
    chunks: list[dict[str, Any]],
) -> dict[str, SearchFn]:
    """
    Monta os buscadores semântico e híbrido.
    """
    try:
        from embeddings import (
            DEFAULT_EMBEDDING_MODEL,
            buscar_chunks_semantico,
            carregar_modelo_embeddings,
            gerar_embeddings_chunks,
        )
        from search_service import buscar_chunks_hibrido
    except Exception as erro:
        raise RuntimeError(
            "Não foi possível carregar as dependências da busca semântica. "
            "Verifique se sentence-transformers está instalado."
        ) from erro

    modelo = carregar_modelo_embeddings(DEFAULT_EMBEDDING_MODEL)
    embeddings_chunks = gerar_embeddings_chunks(
        modelo=modelo,
        chunks=chunks,
    )

    def buscar_semantico(pergunta: str, top_k: int) -> list[dict[str, Any]]:
        return buscar_chunks_semantico(
            pergunta=pergunta,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            modelo=modelo,
            top_k=top_k,
        )

    def buscar_hibrido(pergunta: str, top_k: int) -> list[dict[str, Any]]:
        return buscar_chunks_hibrido(
            pergunta=pergunta,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            modelo=modelo,
            top_k=top_k,
        )

    return {
        "Semântica": buscar_semantico,
        "Híbrida": buscar_hibrido,
    }


def construir_buscadores(
    chunks: list[dict[str, Any]],
    skip_semantic: bool,
) -> dict[str, SearchFn]:
    """
    Monta todos os buscadores disponíveis para a avaliação.
    """
    buscadores: dict[str, SearchFn] = {
        "Lexical": construir_buscador_lexical(chunks),
    }

    if skip_semantic:
        return buscadores

    try:
        buscadores_semanticos = construir_buscadores_semanticos(chunks)
        buscadores.update(buscadores_semanticos)
    except RuntimeError as erro:
        print(f"[aviso] {erro}")
        print("[aviso] A avaliação continuará apenas no modo lexical.")

    return buscadores


def contar_palavras(chunks: list[dict[str, Any]]) -> int:
    """
    Soma a quantidade de palavras presentes nos chunks.
    """
    return sum(len(item["chunk"].split()) for item in chunks)


def preparar_metadados_execucao(
    pdf_path: Path,
    paginas: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    buscadores: dict[str, SearchFn],
) -> dict[str, Any]:
    """
    Monta metadados úteis da execução.
    """
    return {
        "arquivo": pdf_path.name,
        "pdf_path": str(pdf_path.resolve()),
        "paginas_extraidas": len(paginas),
        "chunks_gerados": len(chunks),
        "palavras_nos_chunks": contar_palavras(chunks),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "modos_avaliados": list(buscadores.keys()),
        "casos_de_teste": len(SEARCH_EVAL_CASES),
    }


def obter_keywords_encontradas(
    chunk_texto: str,
    expected_keywords: list[str],
) -> list[str]:
    """
    Retorna quais palavras-chave esperadas foram encontradas no chunk.
    """
    chunk_normalizado = normalizar_texto(chunk_texto)
    keywords_encontradas = []

    for keyword in expected_keywords:
        keyword_normalizada = normalizar_texto(keyword)
        if keyword_normalizada and keyword_normalizada in chunk_normalizado:
            keywords_encontradas.append(keyword)

    return keywords_encontradas


def chunk_atende_caso(
    chunk_resultado: dict[str, Any],
    caso: dict[str, Any],
) -> tuple[bool, list[str]]:
    """
    Verifica se um resultado satisfaz a expectativa do caso de teste.
    """
    pagina_esperada = chunk_resultado["pagina"] in caso["expected_pages"]
    keywords_encontradas = obter_keywords_encontradas(
        chunk_texto=chunk_resultado["chunk"],
        expected_keywords=caso["expected_keywords"],
    )
    quantidade_minima = caso.get("min_keyword_matches", 1)
    keywords_suficientes = len(keywords_encontradas) >= quantidade_minima

    return pagina_esperada and keywords_suficientes, keywords_encontradas


def encontrar_primeiro_rank_relevante(
    resultados: list[dict[str, Any]],
    caso: dict[str, Any],
) -> tuple[int | None, list[str]]:
    """
    Retorna o primeiro rank em que o chunk esperado aparece.
    """
    for posicao, resultado in enumerate(resultados, start=1):
        atende, keywords_encontradas = chunk_atende_caso(resultado, caso)
        if atende:
            return posicao, keywords_encontradas

    return None, []


def resumir_resultados_top(resultados: list[dict[str, Any]]) -> str:
    """
    Resume os primeiros resultados retornados pela busca.
    """
    partes = []

    for posicao, resultado in enumerate(resultados, start=1):
        partes.append(f"#{posicao}: pág. {resultado['pagina']}")

    return " | ".join(partes) if partes else "sem resultados"


def extrair_preview_chunk(
    resultados: list[dict[str, Any]],
    limite: int = 140,
) -> str:
    """
    Extrai um preview do primeiro chunk retornado.
    """
    if not resultados:
        return ""

    texto = resultados[0]["chunk"].replace("\n", " ").strip()
    if len(texto) <= limite:
        return texto

    return texto[:limite].rstrip() + "..."


def avaliar_modo_busca(
    modo: str,
    buscar: SearchFn,
    top_k_max: int,
) -> list[dict[str, Any]]:
    """
    Executa a avaliação completa de um modo de busca.
    """
    linhas_resultado: list[dict[str, Any]] = []

    for caso in SEARCH_EVAL_CASES:
        resultados = buscar(caso["pergunta"], top_k_max)
        rank_relevante, keywords_encontradas = encontrar_primeiro_rank_relevante(
            resultados=resultados,
            caso=caso,
        )

        paginas_retornadas = [resultado["pagina"] for resultado in resultados]

        linha = {
            "modo": modo,
            "case_id": caso["id"],
            "pergunta": caso["pergunta"],
            "expected_pages": ",".join(str(pagina) for pagina in caso["expected_pages"]),
            "expected_keywords": " | ".join(caso["expected_keywords"]),
            "min_keyword_matches": caso.get("min_keyword_matches", 1),
            "rank_relevante": rank_relevante if rank_relevante is not None else "",
            "hit_at_1": int(rank_relevante is not None and rank_relevante <= 1),
            "hit_at_3": int(rank_relevante is not None and rank_relevante <= 3),
            "hit_at_5": int(rank_relevante is not None and rank_relevante <= 5),
            "mrr": round(1 / rank_relevante, 4) if rank_relevante else 0.0,
            "keywords_encontradas": " | ".join(keywords_encontradas),
            "paginas_retornadas": ",".join(str(pagina) for pagina in paginas_retornadas),
            "top_resultados": resumir_resultados_top(resultados),
            "preview_primeiro_resultado": extrair_preview_chunk(resultados),
        }
        linhas_resultado.append(linha)

    return linhas_resultado


def agregar_metricas_por_modo(
    linhas_resultado: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Agrega as métricas finais por modo de busca.
    """
    modos = sorted({linha["modo"] for linha in linhas_resultado})
    resumo: list[dict[str, Any]] = []

    for modo in modos:
        linhas_modo = [linha for linha in linhas_resultado if linha["modo"] == modo]
        total = len(linhas_modo)

        resumo.append(
            {
                "modo": modo,
                "casos": total,
                "hit_at_1": round(sum(linha["hit_at_1"] for linha in linhas_modo) / total, 4),
                "hit_at_3": round(sum(linha["hit_at_3"] for linha in linhas_modo) / total, 4),
                "hit_at_5": round(sum(linha["hit_at_5"] for linha in linhas_modo) / total, 4),
                "mrr": round(sum(linha["mrr"] for linha in linhas_modo) / total, 4),
            }
        )

    return resumo


def salvar_csv(
    caminho_arquivo: Path,
    linhas: list[dict[str, Any]],
) -> None:
    """
    Salva uma lista de dicionários em CSV.
    """
    if not linhas:
        return

    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)

    with caminho_arquivo.open("w", encoding="utf-8", newline="") as arquivo_csv:
        writer = csv.DictWriter(arquivo_csv, fieldnames=list(linhas[0].keys()))
        writer.writeheader()
        writer.writerows(linhas)


def salvar_json(
    caminho_arquivo: Path,
    dados: dict[str, Any],
) -> None:
    """
    Salva um dicionário em JSON.
    """
    caminho_arquivo.parent.mkdir(parents=True, exist_ok=True)
    with caminho_arquivo.open("w", encoding="utf-8") as arquivo_json:
        json.dump(dados, arquivo_json, ensure_ascii=False, indent=2)


def imprimir_resumo(
    metadados_execucao: dict[str, Any],
    resumo_metricas: list[dict[str, Any]],
) -> None:
    """
    Imprime um resumo legível da avaliação.
    """
    print("\n=== AVALIAÇÃO DE BUSCA ===")
    print(f"Arquivo: {metadados_execucao['arquivo']}")
    print(f"Páginas extraídas: {metadados_execucao['paginas_extraidas']}")
    print(f"Chunks gerados: {metadados_execucao['chunks_gerados']}")
    print(
        f"Configuração de chunking: tamanho={metadados_execucao['chunk_size']} | "
        f"overlap={metadados_execucao['overlap']}"
    )
    print(f"Casos de teste: {metadados_execucao['casos_de_teste']}")
    print(f"Modos avaliados: {', '.join(metadados_execucao['modos_avaliados'])}")

    print("\n=== MÉTRICAS FINAIS ===")
    for linha in resumo_metricas:
        print(
            f"{linha['modo']}: "
            f"Hit@1={linha['hit_at_1']:.2%} | "
            f"Hit@3={linha['hit_at_3']:.2%} | "
            f"Hit@5={linha['hit_at_5']:.2%} | "
            f"MRR={linha['mrr']:.4f}"
        )


def imprimir_erros_mais_importantes(
    linhas_resultado: list[dict[str, Any]],
    limite_por_modo: int = 5,
) -> None:
    """
    Mostra os casos em que o resultado esperado não apareceu no top 5.
    """
    modos = sorted({linha["modo"] for linha in linhas_resultado})

    for modo in modos:
        falhas = [
            linha
            for linha in linhas_resultado
            if linha["modo"] == modo and linha["hit_at_5"] == 0
        ]

        if not falhas:
            print(f"\n[{modo}] Nenhuma falha no top 5.")
            continue

        print(f"\n[{modo}] Casos fora do top 5:")
        for linha in falhas[:limite_por_modo]:
            print(
                f"- {linha['case_id']}: {linha['pergunta']} "
                f"| páginas retornadas: {linha['paginas_retornadas'] or 'nenhuma'}"
            )


def executar_avaliacao() -> None:
    """
    Fluxo principal da avaliação.
    """
    args = carregar_argumentos()
    pdf_path = Path(args.pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado: {pdf_path}")

    paginas = carregar_pdf_em_paginas(pdf_path)
    chunks = gerar_chunks_pdf(
        paginas=paginas,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    buscadores = construir_buscadores(
        chunks=chunks,
        skip_semantic=args.skip_semantic,
    )

    metadados_execucao = preparar_metadados_execucao(
        pdf_path=pdf_path,
        paginas=paginas,
        chunks=chunks,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        buscadores=buscadores,
    )

    todas_as_linhas: list[dict[str, Any]] = []

    for modo, buscar in buscadores.items():
        linhas_modo = avaliar_modo_busca(
            modo=modo,
            buscar=buscar,
            top_k_max=args.top_k_max,
        )
        todas_as_linhas.extend(linhas_modo)

    resumo_metricas = agregar_metricas_por_modo(todas_as_linhas)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    salvar_csv(
        caminho_arquivo=output_dir / f"search_eval_results_{timestamp}.csv",
        linhas=todas_as_linhas,
    )
    salvar_csv(
        caminho_arquivo=output_dir / f"search_eval_summary_{timestamp}.csv",
        linhas=resumo_metricas,
    )
    salvar_json(
        caminho_arquivo=output_dir / f"search_eval_run_{timestamp}.json",
        dados={
            "metadados_execucao": metadados_execucao,
            "resumo_metricas": resumo_metricas,
        },
    )

    imprimir_resumo(
        metadados_execucao=metadados_execucao,
        resumo_metricas=resumo_metricas,
    )
    imprimir_erros_mais_importantes(todas_as_linhas)

    print("\nArquivos salvos em:")
    print(output_dir.resolve())


if __name__ == "__main__":
    executar_avaliacao()