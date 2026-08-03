import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
EMBEDDING_MODEL = "models/text-embedding-004"
GENERATION_MODEL = "gemini-2.5-flash"

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
CHROMA_COLLECTION_NAME = "autocorp_knowledge_base"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_CONTEXT_TOKENS = 4000
VECTOR_TOP_K = int(os.getenv("RAG_VECTOR_TOP_K", "10"))
VECTOR_FETCH_K = int(os.getenv("RAG_VECTOR_FETCH_K", "30"))
RERANK_TOP_K = int(os.getenv("RAG_RERANK_TOP_K", "15"))
MIN_VECTOR_SCORE = float(os.getenv("RAG_MIN_VECTOR_SCORE", "0.25"))
MIN_RERANK_SCORE = float(os.getenv("RAG_MIN_RERANK_SCORE", "0.35"))
MIN_GROUNDING_SCORE = float(os.getenv("RAG_MIN_GROUNDING_SCORE", "0.80"))
RRF_K = int(os.getenv("RAG_RRF_K", "60"))

MAX_QUERY_CHARS = int(os.getenv("RAG_MAX_QUERY_CHARS", "2000"))
MAX_UPLOAD_BYTES = int(os.getenv("RAG_MAX_UPLOAD_BYTES", str(10 * 1024 * 1024)))
RAG_MAX_RETRIES = int(os.getenv("RAG_MAX_RETRIES", "3"))
RAG_RETRY_BASE_SECONDS = float(os.getenv("RAG_RETRY_BASE_SECONDS", "0.5"))
RAG_ENABLE_GROUNDING_CHECK = os.getenv(
    "RAG_ENABLE_GROUNDING_CHECK", "true"
).lower() in {"1", "true", "yes"}
RAG_INPUT_COST_PER_MILLION = float(
    os.getenv("RAG_INPUT_COST_PER_MILLION", "0")
)
RAG_OUTPUT_COST_PER_MILLION = float(
    os.getenv("RAG_OUTPUT_COST_PER_MILLION", "0")
)

DOCUMENT_REGISTRY_PATH = os.getenv(
    "KB_DOCUMENT_REGISTRY_PATH",
    os.path.join(CHROMA_PERSIST_DIR, "document_registry.sqlite3"),
)
RAG_TRACE_FILE = os.getenv("RAG_TRACE_FILE", os.path.join("logs", "rag_traces.jsonl"))
KB_DEFAULT_DEPARTMENT = os.getenv("KB_DEFAULT_DEPARTMENT", "general").lower()
KB_DEFAULT_CLEARANCE = os.getenv("KB_DEFAULT_CLEARANCE", "internal").lower()

_genai_configured = False


def _ensure_genai():
    global _genai_configured
    if not _genai_configured:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _genai_configured = True


def get_chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def get_neo4j_driver():
    from neo4j import GraphDatabase
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_gemini_model():
    _ensure_genai()
    import google.generativeai as genai
    return genai.GenerativeModel(GENERATION_MODEL)
