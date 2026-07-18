# AutoCorp Hub

An AI-powered automation platform for corporate workflows — built with Python, Streamlit, LangGraph, Gmail API, Google Calendar API, PostgreSQL, and Google Cloud Storage.

---

## Overview

AutoCorp Hub is a multi-agent automation system orchestrated through **LangGraph** and controlled via a Streamlit dashboard. A LangGraph `StateGraph` fetches unread emails, classifies them by intent, and routes each to the appropriate agent node — handling email replies, meeting scheduling, and HR document requests in a single coordinated pipeline.

---

## Project Structure

```
autocorp-hub/
├── app.py                    # Streamlit dashboard (control plane)
├── orchestrator.py           # LangGraph StateGraph orchestrator
├── mail.py                   # Auto Mail Reply agent
├── meeting_scheduler.py      # Meeting Scheduler agent
├── HR_Document_Request.py    # HR Document Request agent
├── db.py                     # PostgreSQL connection & queries
├── storage_client.py         # Google Cloud Storage client
├── parse_filename.py         # Email subject parser utility
├── agents_config.json        # Runtime config for all agents
├── .env                      # Environment variables
├── credentials.json          # Google OAuth2 credentials
├── token.json                # Gmail OAuth token
├── calendar_token.json       # Google Calendar OAuth token
├── autocorp_storage.json     # GCP service account key
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container image definition
├── k8s/                      # Kubernetes deployment manifests
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── configmap.yaml
│   └── secret.yaml
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

**Modules:**
| File | Purpose |
|------|---------|
| `kb_config.py` | Configuration, API keys, client initialization |
| `kb_vector_store.py` | Chunking, embedding, ChromaDB operations |
| `kb_knowledge_graph.py` | Neo4j entity extraction, storage, graph queries |
| `kb_query_engine.py` | Retrieval, reranking, compression, answer generation |
| `knowledge_base.py` | Streamlit UI page and pipeline orchestration |

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

Each node only processes emails classified for it. The Streamlit dashboard invokes `run_orchestrator(config)` which compiles and runs the graph in a background thread.

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

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy and fill in `.env`:

```env
GMAIL_TOKEN_FILE=token.json
GMAIL_CREDENTIALS_FILE=credentials.json
SENDER_EMAIL=your@email.com

CALENDAR_CREDENTIALS_FILE=credentials.json
CALENDAR_TOKEN_FILE=calendar_token.json

PGHOST=127.0.0.1
PGPORT=5432
PGDATABASE=hr
PGUSER=postgres
PGPASSWORD=yourpassword

GCP_PROJECT_ID=your-gcp-project-id
GCS_BUCKET=your-bucket-name
GOOGLE_APPLICATION_CREDENTIALS=autocorp_storage.json

GMAIL_USER=your@email.com
GMAIL_PROCESSED_LABEL=HR-Auto/Processed

# Knowledge Base
GEMINI_API_KEY=your_gemini_api_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_neo4j_password
CHROMA_PERSIST_DIR=./chroma_data
```

### 3. Set up Google OAuth

Place your `credentials.json` (OAuth2 client) in the project root. On first run, each agent will open a browser window to authorize access and save tokens (`token.json`, `calendar_token.json`).

### 4. Run the dashboard

```bash
streamlit run app.py
```

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
