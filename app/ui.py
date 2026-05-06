import html
import re

import streamlit as st

from app_service import (
    carregar_dados_aplicacao,
    preparar_base_busca,
    executar_busca_documentos,
)
from embeddings import DEFAULT_EMBEDDING_MODEL
from index_storage import INDEX_BASE_DIR
from lexical_search import (
    extrair_termos_relevantes,
    remover_acentos,
)


def configurar_pagina():
    """
    Configura a página principal do app.
    """
    st.set_page_config(page_title="Dev AI Workspace", layout="wide")
    aplicar_estilos_customizados()
    st.title("🚀 Dev AI Workspace")
    st.write("Assistente para leitura, divisão e busca em múltiplos documentos PDF.")


def aplicar_estilos_customizados():
    """
    Aplica estilos visuais simples para melhorar a leitura dos resultados.
    """
    st.markdown(
        """
        <style>
            mark.result-highlight {
                background-color: #fff3a3;
                color: #111827;
                padding: 0.08rem 0.2rem;
                border-radius: 0.25rem;
                font-weight: 600;
            }

            .result-box {
                line-height: 1.7;
                font-size: 1rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def renderizar_sidebar():
    """
    Renderiza a barra lateral com as configurações do app.
    """
    st.sidebar.header("⚙️ Configurações")

    modo_busca = st.sidebar.radio(
        "Modo de busca",
        options=["Híbrida", "Lexical", "Semântica"],
        index=0,
    )

    tamanho_chunk = st.sidebar.slider(
        "Tamanho do chunk (palavras)",
        min_value=100,
        max_value=1000,
        value=500,
        step=50,
    )

    overlap_chunk = st.sidebar.slider(
        "Overlap entre chunks (palavras)",
        min_value=0,
        max_value=200,
        value=80,
        step=10,
    )

    top_k = st.sidebar.slider(
        "Quantidade de resultados da busca",
        min_value=1,
        max_value=15,
        value=5,
        step=1,
    )

    quantidade_chunks_visiveis = st.sidebar.slider(
        "Chunks visíveis por documento",
        min_value=1,
        max_value=20,
        value=5,
        step=1,
    )

    peso_semantico = 0.6
    peso_lexical = 0.4

    if modo_busca == "Híbrida":
        percentual_semantico = st.sidebar.slider(
            "Peso da busca semântica (%)",
            min_value=0,
            max_value=100,
            value=60,
            step=5,
        )
        peso_semantico = percentual_semantico / 100
        peso_lexical = round(1 - peso_semantico, 2)

        st.sidebar.caption(
            f"Peso lexical: {int(peso_lexical * 100)}% | "
            f"Peso semântico: {int(peso_semantico * 100)}%"
        )

    if modo_busca in {"Híbrida", "Semântica"}:
        st.sidebar.caption(
            f"Modelo de embeddings: {DEFAULT_EMBEDDING_MODEL}"
        )

    st.sidebar.caption(
        f"Índices persistentes: {INDEX_BASE_DIR.resolve()}"
    )

    return {
        "modo_busca": modo_busca,
        "tamanho_chunk": tamanho_chunk,
        "overlap_chunk": overlap_chunk,
        "top_k": top_k,
        "quantidade_chunks_visiveis": quantidade_chunks_visiveis,
        "peso_lexical": peso_lexical,
        "peso_semantico": peso_semantico,
    }


def renderizar_upload_pdfs():
    """
    Exibe o campo de upload para múltiplos PDFs.
    """
    return st.file_uploader(
        "Envie um ou mais PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )


def renderizar_resumo_geral(resumo_geral):
    """
    Exibe métricas gerais dos documentos enviados.
    """
    st.subheader("📊 Resumo geral")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("PDFs enviados", resumo_geral["total_documentos"])

    with col2:
        st.metric("Páginas com texto", resumo_geral["total_paginas"])

    with col3:
        st.metric("Palavras extraídas", resumo_geral["total_palavras"])

    with col4:
        st.metric("Chunks gerados", resumo_geral["total_chunks"])


def renderizar_resumo_indexacao(estatisticas_documentos, estatisticas_embeddings):
    """
    Exibe um resumo do uso da persistência local.
    """
    st.subheader("💾 Indexação persistente")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Documentos carregados do disco",
            estatisticas_documentos["documentos_carregados_do_disco"],
        )
        st.metric(
            "Documentos processados agora",
            estatisticas_documentos["documentos_processados_agora"],
        )

    with col2:
        st.metric(
            "Embeddings carregados do disco",
            estatisticas_embeddings["embeddings_carregados_do_disco"],
        )
        st.metric(
            "Embeddings gerados agora",
            estatisticas_embeddings["embeddings_gerados_agora"],
        )

    st.caption(
        "Os índices ficam salvos localmente para evitar reprocessamento e "
        "regeração de embeddings nas próximas execuções."
    )


def renderizar_resumo_documentos(documentos):
    """
    Exibe um resumo individual de cada documento.
    """
    st.subheader("📁 Documentos processados")

    for documento in documentos:
        with st.expander(documento["nome_arquivo"]):
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Páginas", len(documento["paginas"]))

            with col2:
                st.metric("Palavras", len(documento["texto_completo"].split()))

            with col3:
                st.metric("Chunks", len(documento["chunks"]))

            origem = (
                "Índice carregado do disco"
                if documento.get("origem_indice_disco")
                else "Documento processado nesta execução"
            )
            st.caption(origem)


def renderizar_textos_documentos(documentos):
    """
    Exibe o texto completo de cada documento.
    """
    st.subheader("📄 Texto completo dos documentos")

    for documento in documentos:
        with st.expander(f"Texto completo — {documento['nome_arquivo']}"):
            st.text_area(
                label=f"Conteúdo de {documento['nome_arquivo']}",
                value=documento["texto_completo"],
                height=350,
                key=f"texto_{documento['documento_id']}",
            )


def renderizar_chunks_documentos(documentos, quantidade_chunks_visiveis):
    """
    Exibe os chunks gerados por documento.
    """
    st.subheader("🧩 Chunks gerados")

    for documento in documentos:
        with st.expander(f"Chunks — {documento['nome_arquivo']}"):
            chunks_documento = documento["chunks"]

            if not chunks_documento:
                st.warning("Nenhum chunk foi gerado para este documento.")
                continue

            quantidade = min(len(chunks_documento), quantidade_chunks_visiveis)

            for item in chunks_documento[:quantidade]:
                with st.expander(
                    f"Chunk {item['chunk_id_documento']} — Página {item['pagina']}"
                ):
                    st.caption(
                        f"Palavras {item['inicio_palavra']} até {item['fim_palavra']}"
                    )
                    st.write(item["chunk"])


def normalizar_caractere_para_destaque(caractere):
    """
    Normaliza um único caractere sem alterar o comprimento do texto.
    """
    caractere_normalizado = remover_acentos(caractere.lower())

    if not caractere_normalizado:
        return " "

    caractere_base = caractere_normalizado[0]

    if caractere_base.isalnum() or caractere_base == "_":
        return caractere_base

    return " "


def normalizar_texto_para_destaque(texto):
    """
    Normaliza o texto caractere a caractere.
    """
    return "".join(
        normalizar_caractere_para_destaque(caractere)
        for caractere in texto
    )


def mesclar_intervalos(intervalos):
    """
    Une intervalos sobrepostos ou encostados.
    """
    if not intervalos:
        return []

    intervalos_ordenados = sorted(intervalos, key=lambda item: item[0])
    intervalos_mesclados = [intervalos_ordenados[0]]

    for inicio_atual, fim_atual in intervalos_ordenados[1:]:
        ultimo_inicio, ultimo_fim = intervalos_mesclados[-1]

        if inicio_atual <= ultimo_fim:
            intervalos_mesclados[-1] = (
                ultimo_inicio,
                max(ultimo_fim, fim_atual),
            )
        else:
            intervalos_mesclados.append((inicio_atual, fim_atual))

    return intervalos_mesclados


def encontrar_intervalos_destaque(texto_original, termos):
    """
    Encontra os intervalos do texto original que devem ser destacados.
    """
    if not texto_original or not termos:
        return []

    texto_normalizado = normalizar_texto_para_destaque(texto_original)
    intervalos = []

    for termo in termos:
        termo_normalizado = normalizar_texto_para_destaque(termo).strip()

        if not termo_normalizado:
            continue

        padrao = re.compile(rf"\b{re.escape(termo_normalizado)}\b")

        for correspondencia in padrao.finditer(texto_normalizado):
            intervalos.append(
                (
                    correspondencia.start(),
                    correspondencia.end(),
                )
            )

    return mesclar_intervalos(intervalos)


def destacar_termos_no_texto(texto, termos):
    """
    Retorna o texto com os termos destacados em HTML seguro.
    """
    intervalos = encontrar_intervalos_destaque(
        texto_original=texto,
        termos=termos,
    )

    if not intervalos:
        return f"<div class='result-box'>{html.escape(texto)}</div>"

    partes_html = []
    cursor = 0

    for inicio, fim in intervalos:
        if cursor < inicio:
            partes_html.append(html.escape(texto[cursor:inicio]))

        trecho_destacado = html.escape(texto[inicio:fim])
        partes_html.append(
            f"<mark class='result-highlight'>{trecho_destacado}</mark>"
        )
        cursor = fim

    if cursor < len(texto):
        partes_html.append(html.escape(texto[cursor:]))

    return f"<div class='result-box'>{''.join(partes_html)}</div>"


def obter_termos_destaque(resultado, pergunta, modo_busca):
    """
    Define quais termos devem ser usados no destaque visual.
    """
    if modo_busca in {"Lexical", "Híbrida"}:
        termos_encontrados = resultado.get("termos_encontrados", [])
        if termos_encontrados:
            return termos_encontrados

    return extrair_termos_relevantes(pergunta)


def formatar_legenda_score(resultado, modo_busca):
    """
    Formata o score conforme o modo de busca.
    """
    if modo_busca == "Semântica":
        return f"Score semântico: {resultado['score_semantico']}"

    if modo_busca == "Híbrida":
        return (
            f"Score híbrido: {resultado['score_hibrido']} | "
            f"Lexical: {resultado['score_lexical']} | "
            f"Semântico: {resultado['score_semantico']}"
        )

    return f"Score lexical: {resultado['score']}"


def formatar_origem_ranking(resultado, modo_busca):
    """
    Exibe de quais buscas o resultado veio.
    """
    if modo_busca != "Híbrida":
        return None

    origens = resultado.get("origens_ranking", {})
    origens_presentes = []

    if origens.get("lexical"):
        origens_presentes.append("lexical")

    if origens.get("semantico"):
        origens_presentes.append("semântico")

    if not origens_presentes:
        return None

    return "Origem do resultado: " + " + ".join(origens_presentes)


def renderizar_resultados_busca(resultados, pergunta, modo_busca):
    """
    Exibe os resultados da busca com destaque visual.
    """
    if not resultados:
        st.warning("Nenhum trecho relevante encontrado para essa busca.")
        return

    st.success(f"Foram encontrados {len(resultados)} trecho(s) relevante(s).")

    for resultado in resultados:
        termos_destaque = obter_termos_destaque(
            resultado=resultado,
            pergunta=pergunta,
            modo_busca=modo_busca,
        )

        termos_texto = ", ".join(termos_destaque) if termos_destaque else "—"

        chunk_destacado = destacar_termos_no_texto(
            texto=resultado["chunk"],
            termos=termos_destaque,
        )

        with st.expander(
            f"{resultado['arquivo']} | Chunk {resultado['chunk_id_documento']} | Página {resultado['pagina']}"
        ):
            st.caption(
                f"Palavras {resultado['inicio_palavra']} até {resultado['fim_palavra']}"
            )
            st.caption(formatar_legenda_score(resultado, modo_busca))

            origem_ranking = formatar_origem_ranking(resultado, modo_busca)
            if origem_ranking:
                st.caption(origem_ranking)

            st.caption(f"Termos em destaque: {termos_texto}")
            st.markdown(chunk_destacado, unsafe_allow_html=True)


def renderizar_area_busca(
    documentos,
    chunks,
    embeddings_chunks,
    top_k,
    modo_busca,
    peso_lexical,
    peso_semantico,
):
    """
    Exibe a área de busca com filtro por arquivos.
    """
    st.subheader("🔍 Buscar informação nos documentos")

    nomes_arquivos = [documento["nome_arquivo"] for documento in documentos]

    arquivos_selecionados = st.multiselect(
        "Buscar em quais PDFs?",
        options=nomes_arquivos,
        default=nomes_arquivos,
    )

    pergunta = st.text_input(
        "Digite uma pergunta ou palavra-chave:",
        placeholder="Ex: Qual o salário do Analista de Informática?",
    )

    if not pergunta:
        st.info("Digite uma pergunta para buscar os trechos mais relevantes.")
        return

    base_busca = preparar_base_busca(
        chunks=chunks,
        embeddings_chunks=embeddings_chunks,
        arquivos_selecionados=arquivos_selecionados,
        modo_busca=modo_busca,
    )

    chunks_filtrados = base_busca["chunks_filtrados"]
    embeddings_filtrados = base_busca["embeddings_filtrados"]

    if not chunks_filtrados:
        st.warning("Nenhum documento foi selecionado para a busca.")
        return

    if modo_busca in {"Híbrida", "Semântica"}:
        if embeddings_filtrados is None or embeddings_filtrados.size == 0:
            st.warning("Nenhum embedding foi encontrado para os documentos selecionados.")
            return

    with st.spinner("Buscando trechos relevantes..."):
        resultados = executar_busca_documentos(
            pergunta=pergunta,
            chunks=chunks_filtrados,
            embeddings_chunks=embeddings_filtrados,
            top_k=top_k,
            modo_busca=modo_busca,
            peso_lexical=peso_lexical,
            peso_semantico=peso_semantico,
        )

    renderizar_resultados_busca(
        resultados=resultados,
        pergunta=pergunta,
        modo_busca=modo_busca,
    )


def renderizar_estado_inicial():
    """
    Exibe a mensagem inicial antes do upload.
    """
    st.info("📎 Faça upload de um ou mais arquivos PDF para começar.")


def renderizar_erro_processamento(mensagem):
    """
    Exibe mensagem de erro.
    """
    st.error(mensagem)


def main():
    configurar_pagina()
    configuracoes = renderizar_sidebar()

    arquivos_pdf = renderizar_upload_pdfs()

    if not arquivos_pdf:
        renderizar_estado_inicial()
        return

    st.success(f"✅ {len(arquivos_pdf)} arquivo(s) carregado(s) com sucesso!")

    try:
        dados_app = carregar_dados_aplicacao(
            arquivos_pdf=arquivos_pdf,
            tamanho_chunk=configuracoes["tamanho_chunk"],
            overlap_chunk=configuracoes["overlap_chunk"],
            modo_busca=configuracoes["modo_busca"],
        )
    except ValueError as erro:
        renderizar_erro_processamento(str(erro))
        return
    except Exception:
        renderizar_erro_processamento("Não foi possível processar os PDFs enviados.")
        return

    documentos = dados_app["documentos"]
    chunks = dados_app["chunks"]
    embeddings_chunks = dados_app["embeddings_chunks"]
    estatisticas_documentos = dados_app["estatisticas_documentos"]
    estatisticas_embeddings = dados_app["estatisticas_embeddings"]
    resumo_geral = dados_app["resumo_geral"]

    if not documentos:
        st.warning("Não foi possível extrair texto dos PDFs enviados.")
        return

    renderizar_resumo_geral(resumo_geral)

    abas = st.tabs(["Resumo", "Textos", "Chunks", "Busca"])

    with abas[0]:
        renderizar_resumo_indexacao(
            estatisticas_documentos=estatisticas_documentos,
            estatisticas_embeddings=estatisticas_embeddings,
        )
        renderizar_resumo_documentos(documentos)

    with abas[1]:
        renderizar_textos_documentos(documentos)

    with abas[2]:
        renderizar_chunks_documentos(
            documentos=documentos,
            quantidade_chunks_visiveis=configuracoes["quantidade_chunks_visiveis"],
        )

    with abas[3]:
        renderizar_area_busca(
            documentos=documentos,
            chunks=chunks,
            embeddings_chunks=embeddings_chunks,
            top_k=configuracoes["top_k"],
            modo_busca=configuracoes["modo_busca"],
            peso_lexical=configuracoes["peso_lexical"],
            peso_semantico=configuracoes["peso_semantico"],
        )


if __name__ == "__main__":
    main()