import pytest

from chunking import dividir_em_chunks


def test_dividir_em_chunks_gera_chunks_com_metadados():
    paginas = [
        {
            "arquivo": "teste.pdf",
            "pagina": 1,
            "texto": "um dois tres quatro cinco seis sete oito nove dez",
        }
    ]

    chunks = dividir_em_chunks(paginas, tamanho=4, overlap=1)

    assert len(chunks) > 0
    assert chunks[0]["arquivo"] == "teste.pdf"
    assert chunks[0]["pagina"] == 1
    assert chunks[0]["inicio_palavra"] == 0
    assert chunks[0]["fim_palavra"] == 4


def test_dividir_em_chunks_respeita_overlap():
    paginas = [
        {
            "arquivo": "teste.pdf",
            "pagina": 1,
            "texto": "um dois tres quatro cinco seis sete oito",
        }
    ]

    chunks = dividir_em_chunks(paginas, tamanho=4, overlap=2)

    assert chunks[0]["chunk"] == "um dois tres quatro"
    assert chunks[1]["chunk"] == "tres quatro cinco seis"


def test_dividir_em_chunks_rejeita_tamanho_invalido():
    with pytest.raises(ValueError):
        dividir_em_chunks([], tamanho=0, overlap=0)


def test_dividir_em_chunks_rejeita_overlap_maior_ou_igual_tamanho():
    paginas = [{"arquivo": "teste.pdf", "pagina": 1, "texto": "texto simples"}]

    with pytest.raises(ValueError):
        dividir_em_chunks(paginas, tamanho=5, overlap=5)