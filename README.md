# AutoCorp Hub

An AI-powered automation platform for corporate workflows — built with Python, Streamlit, Gmail API, Google Calendar API, PostgreSQL, and Google Cloud Storage.

---

## Overview

AutoCorp Hub is a multi-agent automation system controlled through a Streamlit dashboard. It runs background agents that monitor a Gmail inbox and automatically handle email replies, meeting scheduling, and HR document requests.

---

## Project Structure

```
autocorp-hub/
├── app.py                    # Streamlit dashboard (control plane)
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

---

## Pipeline Flow

```
Streamlit Dashboard (app.py)
    └── agents_config.json
            ├── mail.py
            │     └── Gmail API → filter → auto-reply → mark read
            │
            ├── meeting_scheduler.py
            │     └── Gmail API → detect meeting request
            │           └── Google Calendar API → check conflict → book / notify
            │
            └── HR_Document_Request.py
                  └── Gmail API → validate sender
                        └── PostgreSQL → resolve employee_id
                              └── GCS → fetch file → reply with attachment
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
| Dashboard | Streamlit |
| Email | Gmail API (OAuth2) |
| Calendar | Google Calendar API |
| Database | PostgreSQL (psycopg2) |
| File Storage | Google Cloud Storage |
| Containerization | Docker + Kubernetes |
| Language | Python 3.9+ |
