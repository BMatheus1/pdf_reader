from io import BytesIO

from pypdf import PdfReader


def criar_leitor_pdf(arquivo_bytes):
    """
    Cria um leitor de PDF a partir dos bytes do arquivo.
    """
    return PdfReader(BytesIO(arquivo_bytes))


def extrair_paginas_pdf(arquivo_bytes, nome_arquivo="Documento"):
    """
    Lê um PDF e retorna uma lista com o texto de cada página.

    Exemplo de retorno:
    [
        {"arquivo": "contrato.pdf", "pagina": 1, "texto": "..."},
        {"arquivo": "contrato.pdf", "pagina": 2, "texto": "..."}
    ]
    """
    reader = criar_leitor_pdf(arquivo_bytes)
    paginas = []

    for numero_pagina, pagina in enumerate(reader.pages, start=1):
        texto = pagina.extract_text()

        if texto and texto.strip():
            paginas.append(
                {
                    "arquivo": nome_arquivo,
                    "pagina": numero_pagina,
                    "texto": texto.strip(),
                }
            )

    return paginas


def juntar_texto_paginas(paginas):
    """
    Junta o texto de todas as páginas em uma única string.
    """
    return "\n\n".join(item["texto"] for item in paginas)


def extrair_texto_pdf(arquivo_bytes, nome_arquivo="Documento"):
    """
    Retorna todo o texto do PDF concatenado.
    """
    paginas = extrair_paginas_pdf(
        arquivo_bytes=arquivo_bytes,
        nome_arquivo=nome_arquivo,
    )
    return juntar_texto_paginas(paginas)