import os
import threading
import streamlit as st
from datetime import datetime

from kb_config import CHUNK_SIZE
from kb_vector_store import chunk_document, embed_texts, store_chunks_in_chroma
from kb_knowledge_graph import extract_entities_and_relationships, store_in_neo4j
from kb_query_engine import run_query_pipeline


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


def run_vector_pipeline(text: str, filename: str, status_container):
    try:
        metadata = {
            "source": filename,
            "upload_time": datetime.now().isoformat(),
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


def run_graph_pipeline(text: str, filename: str, status_container):
    try:
        section_size = 6000
        sections = [text[i : i + section_size] for i in range(0, len(text), section_size)]
        total_entities = 0
        total_relationships = 0

        for i, section in enumerate(sections):
            if not section.strip():
                continue
            result = extract_entities_and_relationships(section)
            entities = result.get("entities", [])
            relationships = result.get("relationships", [])
            total_entities += len(entities)
            total_relationships += len(relationships)

            if entities or relationships:
                store_in_neo4j(entities, relationships, filename)

        status_container.write(
            f"  Knowledge Graph: {total_entities} entities, {total_relationships} relationships extracted"
        )
        return True
    except Exception as e:
        status_container.error(f"  Knowledge Graph pipeline error: {e}")
        return False


def run_upload_pipeline(text: str, filename: str, status_container):
    status_container.write(f"**Processing: {filename}**")

    vector_result = [None]
    graph_result = [None]

    def vector_worker():
        vector_result[0] = run_vector_pipeline(text, filename, status_container)

    def graph_worker():
        graph_result[0] = run_graph_pipeline(text, filename, status_container)

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

        uploaded_files = st.file_uploader(
            "Upload PDF, DOCX, TXT, CSV, or MD files",
            type=["pdf", "docx", "txt", "csv", "md"],
            accept_multiple_files=True,
            key="kb_uploader",
        )

        if st.button("Process Documents", type="primary") and uploaded_files:
            progress = st.progress(0)
            status = st.container()

            for i, file in enumerate(uploaded_files):
                text = extract_text_from_file(file)
                if not text.strip():
                    status.warning(f"  {file.name}: No text content found, skipping.")
                    continue

                status.write(f"Extracted {len(text)} characters from {file.name}")
                run_upload_pipeline(text, file.name, status)
                progress.progress((i + 1) / len(uploaded_files))

            st.success(f"Processed {len(uploaded_files)} document(s) successfully!")

    # ─── Query Tab ───
    with tab_query:
        st.subheader("Ask a Question")
        st.write("Query the knowledge base using hybrid retrieval (Vector DB + Knowledge Graph)")

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
                            st.markdown(f"- **[{source_type}]** {source_name}")
                            st.text(src["text"][:300])

        query = st.chat_input("Ask a question about your uploaded documents...")

        if query:
            st.session_state.kb_chat_history.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base (Vector DB + Knowledge Graph)..."):
                    try:
                        result = run_query_pipeline(query)
                        answer = result["answer"]
                        sources = result["sources"]

                        st.markdown(answer)

                        col1, col2 = st.columns(2)
                        col1.metric("Vector Results", result["vector_count"])
                        col2.metric("Graph Triples", result["graph_count"])

                        if sources:
                            with st.expander("Sources & Evidence"):
                                for src in sources:
                                    source_type = src.get("source_type", "")
                                    source_name = src["metadata"].get("source", "Unknown")
                                    st.markdown(f"**[{source_type}]** {source_name}")
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
