# AutoCorp Hub

An AI-powered multi-agent automation platform for corporate workflows — built with Python, Streamlit, LangGraph, Gemini AI, ChromaDB, Neo4j, Gmail API, Google Calendar API, PostgreSQL, and Google Cloud Storage.

---

## Overview

AutoCorp Hub is a multi-agent automation system with two major capabilities:

1. **Email Automation** — Orchestrated through **LangGraph**, a `StateGraph` fetches unread emails, classifies them by intent, and routes each to the appropriate agent node (auto-reply, meeting scheduling, HR document requests).

2. **Knowledge Base (Hybrid RAG)** — Employees can upload documents that get processed through dual pipelines (Vector DB + Knowledge Graph), then ask natural-language questions answered via hybrid retrieval with Gemini 2.5 Flash.

---

## Project Structure

```
autocorp-hub/
├── app.py                    # Streamlit dashboard (control plane)
├── orchestrator.py           # LangGraph StateGraph orchestrator
├── polling_service.py        # Process-wide five-minute Gmail polling service
├── mail.py                   # Auto Mail Reply agent
├── meeting_scheduler.py      # Meeting Scheduler agent
├── HR_Document_Request.py    # HR Document Request agent
├── knowledge_base.py         # Knowledge Base UI + pipeline orchestration
├── kb_config.py              # KB configuration & client initialization
├── kb_vector_store.py        # Chunking, Gemini embeddings, ChromaDB
├── kb_knowledge_graph.py     # Entity extraction, Neo4j graph operations
├── kb_query_engine.py        # Hybrid retrieval, reranking, answer generation
├── db.py                     # PostgreSQL connection & queries
├── storage_client.py         # Google Cloud Storage client
├── parse_filename.py         # Email subject parser utility
├── agents_config.json        # Runtime config for all agents
├── .env                      # Environment variables (not committed)
├── .gitignore                # Git ignore rules
├── credentials.json          # Google OAuth2 credentials (not committed)
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image definition
├── k8s/                      # Kubernetes deployment manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secret.yaml
├── chroma_data/              # ChromaDB persistent storage (auto-generated)
└── logs/                     # Agent log files (auto-generated)
    ├── orchestrator.log
    ├── mail.log
    ├── meeting_scheduler.log
    └── HR_Document_Request.log
```

---

## Agents

### 1. Auto Mail Reply (`mail.py`)
Monitors the inbox for unread emails and sends a canned acknowledgement reply.

- Supports "all emails" or "specific senders" mode
- Marks emails as read after processing
- Logs activity to `logs/mail.log`

### 2. Meeting Scheduler (`meeting_scheduler.py`)
Detects meeting requests in emails and automatically books them on Google Calendar.

- Triggers on emails with subject containing `"schedule a meet"`
- Parses date and time from the email body using regex
- Checks for calendar conflicts before booking
- Replies with the calendar event link or a conflict notice
- Logs activity to `logs/meeting_scheduler.log`

### 3. HR Document Request (`HR_Document_Request.py`)
Allows authorized HR staff to request employee documents via email.

- Only processes emails from a configured allowlist
- Expects subject format: `Request: <filename>` (e.g., `Request: resume.pdf`)
- Looks up the sender's `employee_id` from PostgreSQL
- Fetches the file from GCS bucket under `<employee_id>/<filename>`
- Replies with the file as an email attachment, or suggests available files if not found
- Logs activity to `logs/HR_Document_Request.log`

### 4. Knowledge Base (`knowledge_base.py`)
A hybrid RAG system where employees can upload documents and ask natural-language questions.

**Upload Pipeline (dual):**
- **Vector DB**: Document → Chunking (RecursiveCharacterTextSplitter) → Gemini Embeddings (`text-embedding-004`) → ChromaDB
- **Knowledge Graph**: Document → Entity/Relationship Extraction (Gemini 2.5 Flash) → Neo4j Graph

**Query Pipeline (hybrid):**
1. Query → Gemini embedding → ChromaDB similarity search
2. Query → Entity extraction → Neo4j graph traversal (1-2 hops)
3. Results merged → LLM-based reranking → Context compression
4. Grounded context → Gemini 2.5 Flash → Answer with citations

**Upload Architecture (dual pipeline, parallel):**
```
Employee uploads document (PDF/DOCX/TXT/CSV)
        │
   extract_text_from_file()
        │
        ├──── Thread 1: Vector Pipeline ────────────────────────┐
        │     RecursiveCharacterTextSplitter (1000 chars, 200 overlap)
        │         │                                             │
        │     Gemini text-embedding-004 (batch, 20/request)     │
        │         │                                             │
        │     ChromaDB upsert (cosine similarity)               │
        │                                                       │
        └──── Thread 2: Knowledge Graph Pipeline ───────────────┘
              Split into 6000-char sections
                  │
              Gemini 2.5 Flash → extract entities & relationships (JSON)
                  │
              Neo4j MERGE nodes + CREATE relationships
```

**Query Architecture (hybrid retrieval):**
```
Employee asks a question
        │
   Gemini text-embedding-004 (task_type="retrieval_query")
        │
        ├──── ChromaDB similarity search (top 10 chunks)
        │
        ├──── Neo4j graph traversal:
        │       Gemini extracts entities from query
        │       → fuzzy match nodes → 1-hop outgoing + incoming relationships
        │
   Combine results → LLM-based reranking (Gemini re-scores by relevance)
        │
   Context compression (dedup + token budget)
        │
   Gemini 2.5 Flash → grounded answer with citations
```

**Modules:**
| File | Purpose |
|------|---------|
| `kb_config.py` | Configuration, API keys, lazy client initialization |
| `kb_vector_store.py` | Chunking, Gemini embedding, ChromaDB storage/retrieval |
| `kb_knowledge_graph.py` | Entity/relationship extraction, Neo4j storage, graph traversal |
| `kb_query_engine.py` | Hybrid retrieval, LLM reranking, context compression, answer generation |
| `knowledge_base.py` | Streamlit UI (upload + chat) and pipeline orchestration |

---

## Orchestration (LangGraph)

The orchestrator (`orchestrator.py`) defines a LangGraph `StateGraph` with conditional routing:

```
[START]
   │
   ▼
fetch_emails ──── (no emails?) ───► [END]
   │
   ▼
classify_emails
   │  (rule-based: subject contains "schedule a meet" → meeting,
   │   subject matches "Request: ..." → HR, else → auto-reply)
   │
   ▼
process_meetings
   │  └── Google Calendar API → check conflict → book / notify
   ▼
process_hr_requests
   │  └── PostgreSQL → resolve employee_id → GCS → fetch file → reply with attachment
   ▼
process_auto_replies
   │  └── Gmail API → send canned reply → mark read
   ▼
[END]
```

Each node only processes emails classified for it. After the user saves an active
agent configuration, the Streamlit dashboard starts one process-wide polling
thread. It invokes `run_orchestrator(config)` immediately and then every five
minutes while at least one agent remains active. Saving an updated configuration
wakes the poller immediately, and deactivating all agents stops it. The poller
reloads `agents_config.json` before every run so sender-list changes take effect.
Emails from senders outside a configured allowlist are left unread.

## Pipeline Flow (Legacy)

Individual agents can still be run standalone:

```
python mail.py
python meeting_scheduler.py
python HR_Document_Request.py
```

---

## Setup

### Prerequisites
- Python 3.9+
- PostgreSQL database with an `employees` table
- Google Cloud project with Gmail API, Calendar API, and Cloud Storage enabled
- GCP service account with access to the GCS bucket
- Gemini API key (for Knowledge Base)
- Neo4j instance (local Docker or Aura cloud — for Knowledge Graph RAG)

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy and fill in `.env`:

```env
# Email Agents
GMAIL_TOKEN_FILE=token.json
GMAIL_CREDENTIALS_FILE=credentials.json
SENDER_EMAIL=your@email.com
GMAIL_USER=your@email.com
GMAIL_PROCESSED_LABEL=HR-Auto/Processed
EMAIL_POLL_INTERVAL_SECONDS=300

# Google Calendar
CALENDAR_CREDENTIALS_FILE=credentials.json
CALENDAR_TOKEN_FILE=calendar_token.json

# PostgreSQL
PGHOST=127.0.0.1
PGPORT=5432
PGDATABASE=hr
PGUSER=postgres
PGPASSWORD=yourpassword

# Google Cloud Storage
GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=autocorp_storage.json

# Knowledge Base — Gemini AI
GEMINI_API_KEY=your_gemini_api_key

# Knowledge Base — Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password

# Knowledge Base — ChromaDB
CHROMA_PERSIST_DIR=./chroma_data
```

### 3. Set up Google OAuth

Place your `credentials.json` (OAuth2 client) in the project root. On first run, each agent will open a browser window to authorize access and save tokens (`token.json`, `calendar_token.json`).

### 4. Set up Neo4j (for Knowledge Base)

```bash
docker run -d --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_neo4j_password \
  neo4j:latest
```

Or use [Neo4j Aura](https://neo4j.com/cloud/aura/) (free tier available) and set `NEO4J_URI` to your cloud instance.

### 5. Run the dashboard

```bash
streamlit run app.py
```

Choose the agents and sender lists in the dashboard, then click **Save &
Start/Update Agents**. Gmail is checked immediately and every five minutes after
that. For local testing, `EMAIL_POLL_INTERVAL_SECONDS` can override the default
300-second interval.

---

## Database Schema

The PostgreSQL `hr` database requires an `employees` table:

```sql
CREATE TABLE employees (
    employee_id VARCHAR PRIMARY KEY,
    email       VARCHAR UNIQUE NOT NULL
);
```

---

## GCS Bucket Structure

Employee documents are stored under their `employee_id` as a folder prefix:

```
emp-docs-bucket/
├── EMP001/
│   ├── resume.pdf
│   └── offer_letter.pdf
├── EMP002/
│   └── contract.pdf
```

---

## Deployment

### Docker

```bash
docker build -t autocorp-hub .
docker run --env-file .env -p 8501:8501 autocorp-hub
```

### Kubernetes

```bash
kubectl apply -f k8s/
```

Manifests include `deployment.yaml`, `service.yaml`, `configmap.yaml`, and `secret.yaml`.

---

## Configuration (`agents_config.json`)

The dashboard writes this file on save. You can also edit it manually:

```json
{
    "auto_mail_reply": {
        "active": true,
        "mode": "specific",
        "emails": ["example@company.com"]
    },
    "meeting_scheduler": {
        "active": true,
        "mode": "all",
        "emails": []
    },
    "hr_document_request": {
        "active": true,
        "allowed_emails": ["hr@company.com"]
    }
}
```

- `mode`: `"all"` processes every incoming email, `"specific"` restricts to the listed addresses
- `allowed_emails`: for HR agent, only these senders can request documents

---

## Tech Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph (StateGraph) |
| LLM / Embeddings | Gemini 2.5 Flash + text-embedding-004 |
| Vector Store | ChromaDB |
| Knowledge Graph | Neo4j |
| Dashboard | Streamlit |
| Email | Gmail API (OAuth2) |
| Calendar | Google Calendar API |
| Database | PostgreSQL (psycopg2) |
| File Storage | Google Cloud Storage |
| Containerization | Docker + Kubernetes |
| Language | Python 3.9+ |
