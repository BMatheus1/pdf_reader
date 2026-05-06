def dividir_em_chunks(paginas, tamanho=500, overlap=80):
    """
    Divide o texto em chunks com overlap, preservando:
    - arquivo
    - página
    - posição do trecho
    """
    if tamanho <= 0:
        raise ValueError("O parâmetro 'tamanho' deve ser maior que zero.")

    if overlap < 0:
        raise ValueError("O parâmetro 'overlap' não pode ser negativo.")

    if overlap >= tamanho:
        raise ValueError("O overlap deve ser menor que o tamanho do chunk.")

    if not paginas:
        return []

    passo = tamanho - overlap
    chunks = []
    indice_global = 0

    for item in paginas:
        nome_arquivo = item["arquivo"]
        numero_pagina = item["pagina"]
        texto = item["texto"]

        if not texto or not texto.strip():
            continue

        palavras = texto.split()
        indice_documento = 0

        for inicio in range(0, len(palavras), passo):
            trecho = palavras[inicio:inicio + tamanho]

            if not trecho:
                continue

            chunk_texto = " ".join(trecho).strip()

            if chunk_texto:
                indice_documento += 1

                chunks.append(
                    {
                        "indice": indice_global,
                        "chunk_id_documento": indice_documento,
                        "arquivo": nome_arquivo,
                        "pagina": numero_pagina,
                        "chunk": chunk_texto,
                        "inicio_palavra": inicio,
                        "fim_palavra": inicio + len(trecho),
                    }
                )
                indice_global += 1

            if inicio + tamanho >= len(palavras):
                break

    return chunks