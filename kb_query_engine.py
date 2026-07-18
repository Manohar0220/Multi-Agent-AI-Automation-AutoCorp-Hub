from kb_config import EMBEDDING_MODEL, MAX_CONTEXT_TOKENS, get_gemini_model, _ensure_genai
from kb_vector_store import query_chroma
from kb_knowledge_graph import query_knowledge_graph, format_graph_context


def embed_query(query: str) -> list:
    _ensure_genai()
    import google.generativeai as genai
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query",
    )
    return result["embedding"]


def retrieve_from_vector_store(query_embedding: list, top_k: int = 10) -> list:
    return query_chroma(query_embedding, n_results=top_k)


def retrieve_from_knowledge_graph(query: str) -> list:
    return query_knowledge_graph(query)


def rerank_results(query: str, vector_results: list, graph_triples: list) -> list:
    ranked = []

    for item in vector_results:
        ranked.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "score": item.get("score", 0.5),
            "source_type": "vector",
        })

    if graph_triples:
        graph_context = format_graph_context(graph_triples)
        source_docs = set(t.get("source_doc", "knowledge_graph") for t in graph_triples)
        ranked.append({
            "text": graph_context,
            "metadata": {"source": ", ".join(filter(None, source_docs)), "type": "knowledge_graph"},
            "score": 0.85,
            "source_type": "graph",
        })

    ranked.sort(key=lambda x: x["score"], reverse=True)

    if len(ranked) <= 1:
        return ranked

    model = get_gemini_model()
    candidates = "\n\n".join(
        [f"[{i}] {item['text'][:500]}" for i, item in enumerate(ranked[:15])]
    )
    rerank_prompt = f"""Given the query: "{query}"

Rank these text passages by relevance (most relevant first). Return ONLY a comma-separated list of indices.

Passages:
{candidates}

Ranked indices (most relevant first):"""

    try:
        response = model.generate_content(rerank_prompt)
        raw = response.text.strip()
        indices = [int(x.strip()) for x in raw.split(",") if x.strip().isdigit()]
        reranked = []
        for idx in indices:
            if 0 <= idx < len(ranked):
                reranked.append(ranked[idx])
        for item in ranked:
            if item not in reranked:
                reranked.append(item)
        return reranked
    except Exception:
        return ranked


def compress_context(results: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> str:
    context_parts = []
    current_length = 0

    seen_texts = set()
    for item in results:
        text = item["text"].strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)

        estimated_tokens = len(text) // 4
        if current_length + estimated_tokens > max_tokens:
            remaining = max_tokens - current_length
            if remaining > 100:
                text = text[: remaining * 4]
                context_parts.append(text)
            break

        context_parts.append(text)
        current_length += estimated_tokens

    return "\n\n---\n\n".join(context_parts)


def generate_answer(query: str, context: str) -> str:
    model = get_gemini_model()

    prompt = f"""You are a knowledgeable assistant for AutoCorp. Answer the question based ONLY on the provided context. If the answer is not found in the context, clearly state that.

Context (from knowledge base - includes document chunks and knowledge graph relationships):
---
{context}
---

Question: {query}

Instructions:
- Answer based strictly on the context provided above
- If the context contains relevant relationships from the knowledge graph, incorporate that structured information
- Cite the source document when possible
- If information is insufficient, say so clearly
- Be concise but thorough

Answer:"""

    response = model.generate_content(prompt)
    return response.text


def run_query_pipeline(query: str) -> dict:
    query_embedding = embed_query(query)

    vector_results = retrieve_from_vector_store(query_embedding, top_k=10)
    graph_triples = retrieve_from_knowledge_graph(query)

    ranked = rerank_results(query, vector_results, graph_triples)

    context = compress_context(ranked)

    answer = generate_answer(query, context)

    sources = []
    for item in ranked[:5]:
        sources.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "source_type": item.get("source_type", "unknown"),
        })

    return {
        "answer": answer,
        "sources": sources,
        "context": context,
        "vector_count": len(vector_results),
        "graph_count": len(graph_triples),
    }
