APP_TITLE = "PDF Reader Inteligente"
APP_DESCRIPTION = (
    "Envie um ou mais PDFs, gere índices locais e encontre trechos relevantes "
    "com busca lexical, semântica ou híbrida."
)

SEARCH_MODES = ["Híbrida", "Lexical", "Semântica"]
DEFAULT_SEARCH_MODE = "Híbrida"

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 1000
CHUNK_SIZE_STEP = 50
DEFAULT_CHUNK_SIZE = 500

DEFAULT_CHUNK_OVERLAP = 80
MAX_CHUNK_OVERLAP = 250

DEFAULT_TOP_K = 5
MAX_TOP_K = 15

DEFAULT_VISIBLE_CHUNKS_PER_DOCUMENT = 5
MAX_VISIBLE_CHUNKS_PER_DOCUMENT = 20

DEFAULT_LEXICAL_WEIGHT = 0.4
DEFAULT_SEMANTIC_WEIGHT = 0.6
DEFAULT_DOUBLE_PRESENCE_BONUS = 0.08
DEFAULT_CANDIDATE_MULTIPLIER = 3

DEFAULT_RERANKER_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

DEFAULT_RERANKER_WEIGHT = 0.7
DEFAULT_INITIAL_SEARCH_WEIGHT = 0.3

MIN_CANDIDATE_MULTIPLIER = 2
MAX_CANDIDATE_MULTIPLIER = 6

CUSTOM_CSS = """
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    mark.result-highlight {
        background-color: #fff3a3;
        color: #111827;
        padding: 0.08rem 0.2rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }

    .result-box {
        line-height: 1.75;
        font-size: 1rem;
    }

    .subtle-box {
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-radius: 0.75rem;
        padding: 0.9rem 1rem;
        margin-bottom: 0.75rem;
    }

    .muted-text {
        color: #6b7280;
        font-size: 0.95rem;
    }
</style>
"""