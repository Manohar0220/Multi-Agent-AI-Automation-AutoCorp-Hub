import os
import base64
import re
from email.message import EmailMessage
# Reuse existing Gmail functions
from mail import gmail_service, get_unread_emails, read_email, send_auto_reply, mark_as_read
import json


print("Script started")
from dotenv import load_dotenv
load_dotenv()

from parse_filename import extract_requested_filename
from db import get_employee_id_by_email
from storage_client import fetch_employee_file, list_employee_files


LOG_FILE = "logs/hr_doc_request.log"

# Gmail scopes: read/modify + send
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/gmail.send",
]


CONFIG_FILE = "agents_config.json"
GMAIL_USER = os.getenv("GMAIL_USER")
PROCESSED_LABEL = os.getenv("GMAIL_PROCESSED_LABEL", "HR-Auto/Processed")


if os.path.exists(CONFIG_FILE):
    with open(CONFIG_FILE, "r") as f:
        config = json.load(f)
    hr_conf = config.get("hr_document_request", {})
    ALLOWED_EMAILS = [e.lower() for e in hr_conf.get("allowed_emails", [])]

print(f"Loaded allowed HR emails: {ALLOWED_EMAILS}")

def send_reply_with_attachment(service, to_addr, subject, body_text, attachment_bytes, filename, mime):
    """Send reply email with attachment."""
    msg = EmailMessage()
    msg["To"] = to_addr
    msg["From"] = GMAIL_USER
    msg["Subject"] = f"Re: {subject or '(No Subject)'}"
    msg.set_content(body_text)

    maintype, subtype = (mime.split("/", 1) if "/" in mime else ("application", "octet-stream"))
    msg.add_attachment(attachment_bytes, maintype=maintype, subtype=subtype, filename=filename)

    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
    body = {"raw": raw}
    service.users().messages().send(userId="me", body=body).execute()
    print(f" Sent {filename} to {to_addr}")




# ---------------- Main processing ----------------

def process_document_requests():
    print("entered document request function")
    service = gmail_service()
    print("getting unread emails")
    messages = get_unread_emails(service)
    if not messages:
        print(" No unread emails found.")
        return

    print(f" Checking unread mails for HR document requests...")
    for msg in messages:
        try:
            msg_id = msg["id"]
            subject, sender, body = read_email(service, msg_id)
            print(f"--- Processing Message ---")
            print(f"Subject: {subject}")
            print(f"Sender: {sender}")

            if not subject or not sender:
                continue

            sender_email = sender.split("<")[-1].replace(">", "").strip().lower()
            if sender_email not in ALLOWED_EMAILS:
                print(f"Skipping {sender_email} — not in allowed HR sender list.")
                mark_as_read(service, msg_id)
                continue

            requested = extract_requested_filename(subject)
            print(f" Requested file from {sender_email}: {requested}")

            # If subject invalid
            if not requested:
                send_auto_reply(
                    service,
                    sender_email,
                    subject,
                    "Please use subject format: 'Request: <file name>'. Example: Request: resume.pdf",
                )
                mark_as_read(service, msg_id)
                continue

            # Lookup employee_id in DB
            employee_id = get_employee_id_by_email(sender_email)
            if not employee_id:
                send_auto_reply(
                    service,
                    sender_email,
                    subject,
                    "Your email was not found in our HR database. Please use your registered company email.",
                )
                mark_as_read(service, msg_id)
                continue

            # Fetch document from GCS
            file_obj = fetch_employee_file(employee_id, requested)
            if not file_obj:
                suggestions = list_employee_files(employee_id, limit=10)
                suggestion_text = ", ".join(suggestions) if suggestions else "None"
                send_auto_reply(
                    service,
                    sender_email,
                    subject,
                    f"Could not find '{requested}' in your HR folder ({employee_id}).\n"
                    f"Available files: {suggestion_text}",
                )
                mark_as_read(service, msg_id)
                continue

            data, mime = file_obj
            send_reply_with_attachment(
                service,
                sender_email,
                subject,
                f"Hi, attaching '{requested}' as requested.",
                data,
                requested,
                mime,
            )

            mark_as_read(service, msg_id)
            print(f" Replied to {sender_email} with '{requested}'")

        except Exception as e:
            print(f" Error processing message: {e}")
            continue


# ---------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    process_document_requests()
 