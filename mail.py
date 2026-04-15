import os
import base64
import time
import json
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from dotenv import load_dotenv
import sys
from google.auth.transport.requests import Request
import datetime

load_dotenv()

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
CONFIG_FILE = "agents_config.json"

LOG_FILE = "logs/mail.log"
os.makedirs("logs", exist_ok=True)
if __name__ == "__main__":
    sys.stdout = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
# ------------------------------------------------------------------
# Gmail Service Authentication
# ------------------------------------------------------------------
def gmail_service():
    creds = None
    token_file = os.getenv("GMAIL_TOKEN_FILE")
    creds_file = os.getenv("GMAIL_CREDENTIALS_FILE")

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)

        #  Auto-refresh logic
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif creds and creds.expiry and creds.expiry < datetime.datetime.utcnow() + datetime.timedelta(minutes=5):
            creds.refresh(Request())

        # Save updated creds if refreshed
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    else:
        flow = InstalledAppFlow.from_client_secrets_file(creds_file, SCOPES)
        flow.redirect_uri = "http://localhost:8080/"
        creds = flow.run_local_server(port=8080)
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("gmail", "v1", credentials=creds)


# ------------------------------------------------------------------
# Gmail Utility Functions
# ------------------------------------------------------------------
def get_unread_emails(service):
    results = service.users().messages().list(userId='me', labelIds=['UNREAD'], maxResults=10).execute()
    return results.get('messages', [])


def read_email(service, msg_id):
    msg = service.users().messages().get(userId='me', id=msg_id, format='full').execute()
    payload = msg['payload']
    headers = payload.get('headers', [])
    subject = sender = None
    for h in headers:
        if h['name'] == 'Subject':
            subject = h['value']
        if h['name'] == 'From':
            sender = h['value']
    body = ""
    if 'data' in payload.get('body', {}):
        body = base64.urlsafe_b64decode(payload['body']['data']).decode('utf-8', errors='ignore')
    else:
        for part in payload.get('parts', []):
            if part.get('mimeType') == 'text/plain' and 'data' in part['body']:
                body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8', errors='ignore')
    return subject, sender, body


def send_auto_reply(service, to, subject, body):
    message_text = f"Hello,\n\n{body}\n\nRegards,\nAutoCorp Hub"
    message = MIMEText(message_text)
    message['to'] = to
    message['subject'] = f"Re: {subject}"
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    service.users().messages().send(userId='me', body={'raw': raw}).execute()
    print(f" Auto-reply sent to: {to}")


def mark_as_read(service, msg_id):
    service.users().messages().modify(
        userId='me', id=msg_id, body={'removeLabelIds': ['UNREAD']}
    ).execute()


# ------------------------------------------------------------------
# Main Agent Logic
# ------------------------------------------------------------------
def main():
    # Load agent configuration
    if not os.path.exists(CONFIG_FILE):
        print(" Config file not found. Run Streamlit dashboard first.")
        return
    config = json.load(open(CONFIG_FILE))

    agent_conf = config.get("auto_mail_reply", {})
    if not agent_conf.get("active", False):
        print(" Auto Mail Reply Agent is inactive.")
        return

    specific_mode = agent_conf.get("mode") == "specific"
    allowed_emails = [e.lower() for e in agent_conf.get("emails", [])]

    service = gmail_service()
    messages = get_unread_emails(service)
    if not messages:
        print("No new unread emails.")
        return

    for msg in messages:
        msg_id = msg["id"]
        subject, sender, body = read_email(service, msg_id)
        if not sender:
            continue

        print(f"\n New Email from {sender} — Subject: {subject}")
        sender_email = sender.split("<")[-1].replace(">", "").strip().lower()

        # --- Filter according to configuration
        if (not specific_mode) or (sender_email in allowed_emails):
            print(f" Auto-reply triggered for {sender_email}")
            reply_body = "Thank you for your email. Our team will respond shortly."
            send_auto_reply(service, sender_email, subject or "(No Subject)", reply_body)
        else:
            print(f" Skipping email from {sender_email} (not in target list)")

        mark_as_read(service, msg_id)
        time.sleep(2)


if __name__ == "__main__":
    main()
