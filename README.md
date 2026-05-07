# PDF Reader Inteligente

Aplicação em **Streamlit** para leitura e busca em PDFs com foco em uso prático no dia a dia.

O projeto permite enviar um ou mais arquivos PDF, gerar índices locais e pesquisar trechos relevantes usando três modos de busca:

- **Lexical**: encontra termos e frases presentes no texto
- **Semântica**: encontra trechos parecidos em significado
- **Híbrida**: combina busca lexical e semântica para melhorar relevância

---

## Objetivo do projeto

Este projeto foi construído para ser um **leitor inteligente de PDFs**, útil para:

- estudar materiais extensos
- localizar rapidamente cláusulas, trechos e informações importantes
- pesquisar contratos, relatórios, artigos e documentos longos
- servir como base para uma futura aplicação de RAG

O foco é unir:

- **código limpo**
- **boa manutenção**
- **experiência de uso clara**
- **valor real no dia a dia**

---

## Funcionalidades

- Upload de **múltiplos PDFs**
- Extração de texto por página
- Divisão em **chunks com overlap**
- Busca **lexical**
- Busca **semântica com embeddings**
- Busca **híbrida com reranking**
- Exibição da **origem do trecho**
- Filtro por arquivo
- Cache persistente de:
  - documentos processados
  - embeddings por documento
- Resumo dos documentos processados
- Interface em abas para busca e inspeção dos PDFs

---

## Estrutura do projeto

```txt
pdf_reader/
├── app/
│   ├── __init__.py
│   ├── app_service.py
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluate_search.py
│   ├── index_storage.py
│   ├── ingest.py
│   ├── lexical_search.py
│   ├── search_eval_cases.py
│   ├── search_service.py
│   ├── ui.py
│   ├── ui_highlight.py
│   ├── ui_results.py
│   └── ui_sidebar.py
├── tests/
│   ├── conftest.py
│   ├── test_chunking.py
│   ├── test_lexical_search.py
│   └── test_search_service.py
├── .gitignore
├── README.md
└── requirements.txt
```

# Tecnologias utilizadas

- Python
- Streamlit
- PyPDF
- NumPy
- Sentence Transformers
- Pytest

## Como executar

### 1. Criar e ativar ambiente virtual

No Windows PowerShell:

```bash 
python -m venv .venv
.venv\Scripts\Activate.ps1
```

2. Instalar dependências

```bash
pip install -r requirements.txt
```

3. Rodar a aplicação

```bash
streamlit run app/ui.py
```

## Como usar

- Abra a aplicação
- Envie um ou mais PDFs
- Escolha o modo de busca:
- Lexical
- Semântica
- Híbrida
- Digite uma pergunta ou termo
- Analise os trechos encontrados, com página e arquivo de origem

## Exemplos de busca:

- o que o documento diz sobre estabilidade
- onde fala de plano de saúde
- qual é o prazo citado
- quais direitos foram mencionados

## Cache e persistência local

O projeto salva índices locais para evitar reprocessamento desnecessário.

São persistidos:

- documentos já processados
- embeddings por documento
- metadados dos embeddings

Os arquivos ficam em:
```bash
.dev_ai_workspace_index/
```

Isso melhora a velocidade de uso nas próximas execuções.

## Qualidade do código

O projeto foi organizado com foco em:

- separação de responsabilidades
- funções pequenas e claras
- nomes de funções descritivos
- estrutura simples para manutenção
- base preparada para evoluir sem virar bagunça
- Próximas melhorias
- melhorar o ranking híbrido
- exibir prévia por página com navegação melhor
- adicionar filtros mais avançados
- permitir exportação dos resultados
- preparar resposta baseada em contexto recuperado
- evoluir para um fluxo de RAG

## Autor

- Projeto desenvolvido por Matheus Brito da Silva

## GitHub:
- https://github.com/BMatheus1