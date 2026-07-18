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
