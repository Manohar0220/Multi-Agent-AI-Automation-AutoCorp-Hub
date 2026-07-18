import hashlib
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

from kb_config import (
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    CHROMA_COLLECTION_NAME,
    get_chroma_client,
    _ensure_genai,
)


def chunk_document(text: str, metadata: dict) -> list:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    splits = splitter.split_text(text)
    chunks = []
    for i, chunk_text in enumerate(splits):
        chunks.append({
            "text": chunk_text,
            "metadata": {
                **metadata,
                "chunk_index": i,
                "total_chunks": len(splits),
            },
        })
    return chunks


def embed_texts(texts: list, task_type: str = "retrieval_document") -> list:
    _ensure_genai()
    import google.generativeai as genai
    embeddings = []
    batch_size = 20
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=batch,
            task_type=task_type,
        )
        embeddings.extend(result["embedding"])
        if i + batch_size < len(texts):
            time.sleep(0.5)
    return embeddings


def store_chunks_in_chroma(chunks: list, embeddings: list):
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    ids = []
    documents = []
    metadatas = []
    for chunk in chunks:
        chunk_id = hashlib.md5(chunk["text"].encode()).hexdigest()
        ids.append(chunk_id)
        documents.append(chunk["text"])
        metadatas.append(chunk["metadata"])

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )
    return len(ids)


def query_chroma(query_embedding: list, n_results: int = 10) -> list:
    client = get_chroma_client()
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    output = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            output.append({
                "text": doc,
                "metadata": meta,
                "score": 1 - dist,
            })
    return output
