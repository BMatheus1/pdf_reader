# PDF Reader Inteligente

Aplicação web em **Streamlit** para pesquisar informações em **um ou mais PDFs** com foco em uso prático no dia a dia.

O sistema processa os documentos, gera índices locais e permite buscar trechos relevantes em três modos:

- **Lexical**: busca termos exatos e palavras presentes no texto
- **Semântica**: busca trechos parecidos em significado
- **Híbrida**: combina busca textual e semântica para melhorar relevância

---

## Visão geral

O objetivo do projeto é transformar documentos longos em uma base consultável de forma rápida, clara e rastreável.

A aplicação foi pensada para cenários como:

- leitura de contratos
- consulta de cláusulas e prazos
- análise de relatórios e documentos extensos
- estudo de materiais em PDF
- recuperação de informação com evidência de origem

Além da busca, o app entrega:

- **resposta curta em linguagem natural**
- **fontes usadas na resposta**
- **trechos de apoio destacados**
- **página e arquivo de origem**
- **cache local para evitar reprocessamento**

---

## Funcionalidades

- Upload de **múltiplos PDFs**
- Extração de texto por página
- Geração de **chunks com overlap**
- Busca **lexical**
- Busca **semântica com embeddings**
- Busca **híbrida**
- **Reranking** dos melhores candidatos
- Resposta curta com **evidências rastreáveis**
- Filtro por arquivo
- Cache persistente de:
  - documentos processados
  - embeddings
  - metadados de indexação
- Interface com abas para:
  - busca
  - inspeção dos documentos processados

---

## Arquitetura do projeto

```txt
pdf_reader/
├── app/
│   ├── __init__.py
│   ├── answer_service.py
│   ├── app_service.py
│   ├── chunking.py
│   ├── config.py
│   ├── embeddings.py
│   ├── evaluate_search.py
│   ├── index_storage.py
│   ├── ingest.py
│   ├── lexical_search.py
│   ├── query_understanding.py
│   ├── reranking_service.py
│   ├── search_eval_cases.py
│   ├── search_service.py
│   ├── ui.py
│   ├── ui_highlight.py
│   ├── ui_results.py
│   └── ui_sidebar.py
├── tests/
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