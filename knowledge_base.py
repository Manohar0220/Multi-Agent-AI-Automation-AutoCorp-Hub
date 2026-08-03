from __future__ import annotations

import threading
import streamlit as st
from datetime import datetime

from kb_config import KB_DEFAULT_CLEARANCE, KB_DEFAULT_DEPARTMENT
from kb_vector_store import chunk_document, embed_texts, store_chunks_in_chroma
from kb_knowledge_graph import extract_entities_and_relationships, store_in_neo4j
from kb_query_engine import run_query_pipeline
from kb_document_registry import register_document, update_document_status
from kb_guardrails import validate_upload


def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()

    if filename.endswith(".txt") or filename.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="ignore")

    elif filename.endswith(".pdf"):
        import PyPDF2
        import io
        reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text

    elif filename.endswith(".docx"):
        import docx
        import io
        doc = docx.Document(io.BytesIO(uploaded_file.read()))
        return "\n".join(para.text for para in doc.paragraphs if para.text.strip())

    elif filename.endswith(".csv"):
        import pandas as pd
        import io
        df = pd.read_csv(io.BytesIO(uploaded_file.read()))
        return df.to_string(index=False)

    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")


def run_vector_pipeline(
    text: str, filename: str, status_container, document_metadata: dict | None = None
):
    try:
        metadata = {
            "source": filename,
            "upload_time": datetime.now().isoformat(),
            **(document_metadata or {}),
        }
        chunks = chunk_document(text, metadata)
        status_container.write(f"  Chunked into {len(chunks)} segments")

        texts = [c["text"] for c in chunks]
        embeddings = embed_texts(texts, task_type="retrieval_document")
        status_container.write(f"  Generated {len(embeddings)} embeddings")

        count = store_chunks_in_chroma(chunks, embeddings)
        status_container.write(f"  Stored {count} chunks in ChromaDB")
        return True
    except Exception as e:
        status_container.error(f"  Vector pipeline error: {e}")
        return False


def run_graph_pipeline(
    text: str, filename: str, status_container, document_metadata: dict | None = None
):
    try:
        section_size = 6000
        sections = [text[i : i + section_size] for i in range(0, len(text), section_size)]
        total_entities = 0
        total_relationships = 0

        for i, section in enumerate(sections):
            if not section.strip():
                continue
            result = extract_entities_and_relationships(section)
            if result.get("error"):
                raise RuntimeError(
                    f"Entity extraction failed in section {i + 1}: {result['error']}"
                )
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            total_entities += len(entities)
            total_relationships += len(relationships)

            if entities or relationships:
                store_in_neo4j(
                    entities,
                    relationships,
                    filename,
                    document_metadata=document_metadata,
                )

        status_container.write(
            f"  Knowledge Graph: {total_entities} entities, {total_relationships} relationships extracted"
        )
        return True
    except Exception as e:
        status_container.error(f"  Knowledge Graph pipeline error: {e}")
        return False


def run_upload_pipeline(
    text: str, filename: str, status_container, document_metadata: dict | None = None
):
    status_container.write(f"**Processing: {filename}**")

    vector_result = [None]
    graph_result = [None]

    def vector_worker():
        vector_result[0] = run_vector_pipeline(
            text, filename, status_container, document_metadata
        )

    def graph_worker():
        graph_result[0] = run_graph_pipeline(
            text, filename, status_container, document_metadata
        )

    t1 = threading.Thread(target=vector_worker)
    t2 = threading.Thread(target=graph_worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    return vector_result[0], graph_result[0]


def render_knowledge_base_page():
    st.title("Knowledge Base")
    st.caption("Upload documents and ask questions — powered by hybrid Vector + Knowledge Graph RAG")

    tab_upload, tab_query = st.tabs(["Upload Documents", "Ask Questions"])

    # ─── Upload Tab ───
    with tab_upload:
        st.subheader("Upload Documents")
        st.write("Upload files to build the knowledge base. Documents are processed through two pipelines:")
        col1, col2 = st.columns(2)
        with col1:
            st.info("**Vector Pipeline**\n\nChunking → Gemini Embeddings → ChromaDB")
        with col2:
            st.info("**Knowledge Graph Pipeline**\n\nEntity Extraction → Neo4j Graph")

        metadata_col1, metadata_col2, metadata_col3 = st.columns(3)
        with metadata_col1:
            document_owner = st.text_input("Document owner", value="knowledge-admin")
        with metadata_col2:
            document_department = st.text_input("Department", value="general")
        with metadata_col3:
            document_classification = st.selectbox(
                "Classification",
                ["public", "internal", "confidential", "restricted"],
                index=1,
            )

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, CSV, or MD files",
            type=["pdf", "docx", "txt", "csv", "md"],
            accept_multiple_files=True,
            key="kb_uploader",
        )

        if st.button("Process Documents", type="primary") and uploaded_files:
            progress = st.progress(0)
            status = st.container()
            successful_documents = 0

            for i, file in enumerate(uploaded_files):
                text = extract_text_from_file(file)
                document = register_document(
                    filename=file.name,
                    text=text,
                    owner=document_owner,
                    department=document_department.strip().lower() or "general",
                    classification=document_classification,
                )
                violations = validate_upload(
                    file.name, int(getattr(file, "size", 0) or 0), text
                )
                if violations:
                    update_document_status(document["document_id"], "quarantined")
                    status.error(
                        f"{file.name} was quarantined: " + "; ".join(violations)
                    )
                    progress.progress((i + 1) / len(uploaded_files))
                    continue

                status.write(f"Extracted {len(text)} characters from {file.name}")
                vector_ok, graph_ok = run_upload_pipeline(
                    text, file.name, status, document_metadata=document
                )
                if vector_ok and graph_ok:
                    final_status = "indexed"
                    successful_documents += 1
                elif vector_ok or graph_ok:
                    final_status = "partially_indexed"
                else:
                    final_status = "failed"
                update_document_status(document["document_id"], final_status)
                progress.progress((i + 1) / len(uploaded_files))

            if successful_documents:
                st.success(
                    f"Successfully indexed {successful_documents} of {len(uploaded_files)} document(s)."
                )
            else:
                st.warning("No documents completed both ingestion pipelines.")

    # ─── Query Tab ───
    with tab_query:
        st.subheader("Ask a Question")
        st.write("Query the knowledge base using hybrid retrieval (Vector DB + Knowledge Graph)")

        access_claims = st.session_state.get(
            "kb_access_claims",
            {
                "department": KB_DEFAULT_DEPARTMENT,
                "clearance": KB_DEFAULT_CLEARANCE,
                "allow_pii": False,
            },
        )
        access_department = access_claims.get("department", KB_DEFAULT_DEPARTMENT)
        access_clearance = access_claims.get("clearance", KB_DEFAULT_CLEARANCE)
        st.caption(
            "Access scope: "
            f"department={access_department}, "
            f"clearance={access_clearance}. "
            "Production deployments should populate these claims from authenticated SSO/RBAC."
        )

        if "kb_chat_history" not in st.session_state:
            st.session_state.kb_chat_history = []

        for msg in st.session_state.kb_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander("Sources"):
                        for src in msg["sources"]:
                            source_type = src.get("source_type", "")
                            source_name = src["metadata"].get("source", "Unknown")
                            source_id = src.get("source_id", "")
                            st.markdown(
                                f"- **[{source_type}]** {source_name} — `{source_id}`"
                            )
                            st.text(src["text"][:300])

        query = st.chat_input("Ask a question about your uploaded documents...")

        if query:
            st.session_state.kb_chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base (Vector DB + Knowledge Graph)..."):
                    try:
                        result = run_query_pipeline(
                            query,
                            access_context={
                                "department": access_department,
                                "clearance": access_clearance,
                                "allow_pii": bool(access_claims.get("allow_pii", False)),
                            },
                        )
                        answer = result["answer"]
                        sources = result["sources"]

                        if result.get("blocked"):
                            st.error(answer)
                        elif result.get("abstained"):
                            st.warning(answer)
                        else:
                            st.markdown(answer)

                        col1, col2, col3 = st.columns(3)
                        col1.metric("Vector Results", result["vector_count"])
                        col2.metric("Graph Triples", result["graph_count"])
                        total_latency = sum(result.get("latencies_ms", {}).values())
                        col3.metric("Measured Stage Time", f"{total_latency:.0f} ms")

                        if result.get("citations"):
                            st.caption(
                                "Verified citations: " + ", ".join(result["citations"])
                            )
                            st.caption(
                                f"Grounding score: {result.get('grounding_score', 0):.2f} · "
                                f"Trace: {result.get('trace_id', '')}"
                            )
                        if result.get("estimated_cost_usd"):
                            st.caption(
                                f"Estimated generation cost: ${result['estimated_cost_usd']:.6f}"
                            )
                        if result.get("pii_redacted"):
                            st.info(
                                "Sensitive output was redacted: "
                                + ", ".join(result["pii_redacted"])
                            )
                        if result.get("errors"):
                            st.info("Fallbacks used: " + "; ".join(result["errors"]))

                        if sources:
                            with st.expander("Sources & Evidence"):
                                for src in sources:
                                    source_type = src.get("source_type", "")
                                    source_name = src["metadata"].get("source", "Unknown")
                                    st.markdown(
                                        f"**[{source_type}]** {source_name} — `{src.get('source_id', '')}`"
                                    )
                                    st.caption(
                                        f"Rerank relevance: {src.get('rerank_score', 0):.3f}"
                                    )
                                    st.text(src["text"][:300])
                                    st.divider()

                        st.session_state.kb_chat_history.append({
                            "role": "assistant",
                            "content": answer,
                            "sources": sources,
                        })

                    except Exception as e:
                        error_msg = f"Error processing query: {e}"
                        st.error(error_msg)
                        st.session_state.kb_chat_history.append({
                            "role": "assistant",
                            "content": error_msg,
                        })

        if st.session_state.kb_chat_history and st.button("Clear Chat History"):
            st.session_state.kb_chat_history = []
            st.rerun()
