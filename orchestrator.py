import json
import os
import re
import logging
from typing import TypedDict, List, Dict, Any

from langgraph.graph import StateGraph, END

from mail import gmail_service, get_unread_emails, read_email, send_auto_reply, mark_as_read
from meeting_scheduler import (
    google_calendar_service,
    parse_meeting_datetime,
    check_calendar_conflict,
    schedule_calendar_event,
)
from HR_Document_Request import send_reply_with_attachment, load_allowed_emails
from parse_filename import extract_requested_filename
from db import get_employee_id_by_email
from storage_client import fetch_employee_file, list_employee_files

CONFIG_FILE = "agents_config.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("orchestrator")
logging.basicConfig(
    filename=os.path.join(LOG_DIR, "orchestrator.log"),
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


# ────────────────────────────────────────────────────────────────────
# State schema
# ────────────────────────────────────────────────────────────────────

class EmailItem(TypedDict):
    msg_id: str
    subject: str
    sender: str
    sender_email: str
    body: str


class OrchestratorState(TypedDict):
    config: Dict[str, Any]
    emails: List[EmailItem]
    classified: Dict[str, List[EmailItem]]
    results: List[str]


# ────────────────────────────────────────────────────────────────────
# Graph nodes
# ────────────────────────────────────────────────────────────────────

def fetch_emails_node(state: OrchestratorState) -> dict:
    service = gmail_service()
    raw_messages = get_unread_emails(service)
    emails = []
    for msg in raw_messages:
        msg_id = msg["id"]
        subject, sender, body = read_email(service, msg_id)
        if not sender:
            continue
        sender_email = sender.split("<")[-1].replace(">", "").strip().lower()
        emails.append(EmailItem(
            msg_id=msg_id,
            subject=subject or "",
            sender=sender,
            sender_email=sender_email,
            body=body or "",
        ))
    logger.info(f"Fetched {len(emails)} unread emails")
    return {"emails": emails, "results": [f"Fetched {len(emails)} unread emails"]}


def classify_emails_node(state: OrchestratorState) -> dict:
    classified: Dict[str, List[EmailItem]] = {"meeting": [], "hr": [], "auto_reply": []}
    config = state["config"]

    for email in state["emails"]:
        subj_lower = email["subject"].lower()
        if "schedule a meet" in subj_lower:
            classified["meeting"].append(email)
        elif re.match(r"\s*request:\s*.+", subj_lower):
            classified["hr"].append(email)
        else:
            classified["auto_reply"].append(email)

    summary = (
        f"Classified: {len(classified['meeting'])} meeting, "
        f"{len(classified['hr'])} HR, {len(classified['auto_reply'])} auto-reply"
    )
    logger.info(summary)
    return {"classified": classified, "results": state["results"] + [summary]}


def process_meetings_node(state: OrchestratorState) -> dict:
    config = state["config"]
    agent_conf = config.get("meeting_scheduler", {})
    results = list(state["results"])

    if not agent_conf.get("active", False):
        results.append("Meeting Scheduler agent is inactive — skipped")
        return {"results": results}

    meeting_emails = state["classified"].get("meeting", [])
    if not meeting_emails:
        return {"results": results}

    specific_mode = agent_conf.get("mode") == "specific"
    allowed = [e.lower() for e in agent_conf.get("emails", [])]

    gmail = gmail_service()
    calendar = google_calendar_service()

    for email in meeting_emails:
        try:
            if specific_mode and email["sender_email"] not in allowed:
                mark_as_read(gmail, email["msg_id"])
                results.append(f"Meeting: skipped {email['sender_email']} (not in target list)")
                continue

            meeting_times = parse_meeting_datetime(email["body"])
            if not meeting_times:
                mark_as_read(gmail, email["msg_id"])
                results.append(f"Meeting: could not parse date/time from {email['sender_email']}")
                continue

            meeting_start, meeting_end = meeting_times
            conflict = check_calendar_conflict(calendar, meeting_start, meeting_end)

            if conflict:
                reply = f"Hi, I already have an event at {meeting_start.strftime('%I:%M %p, %b %d %Y')}."
            else:
                event = schedule_calendar_event(calendar, "Auto-Scheduled Meeting", meeting_start, meeting_end)
                reply = (
                    f"Your meeting has been scheduled for {meeting_start.strftime('%I:%M %p, %b %d %Y')}.\n\n"
                    f"Event: {event.get('htmlLink')}"
                )

            send_auto_reply(gmail, email["sender_email"], email["subject"], reply)
            mark_as_read(gmail, email["msg_id"])
            results.append(f"Meeting: processed {email['sender_email']} — {'conflict' if conflict else 'scheduled'}")
        except Exception as e:
            results.append(f"Meeting: error for {email['sender_email']}: {e}")
            logger.exception(f"Error in meeting node for {email['msg_id']}")

    return {"results": results}


def process_hr_node(state: OrchestratorState) -> dict:
    config = state["config"]
    agent_conf = config.get("hr_document_request", {})
    results = list(state["results"])

    if not agent_conf.get("active", False):
        results.append("HR Document Request agent is inactive — skipped")
        return {"results": results}

    hr_emails = state["classified"].get("hr", [])
    if not hr_emails:
        return {"results": results}

    allowed_emails = [e.lower() for e in agent_conf.get("allowed_emails", [])]
    gmail = gmail_service()

    for email in hr_emails:
        try:
            if email["sender_email"] not in allowed_emails:
                mark_as_read(gmail, email["msg_id"])
                results.append(f"HR: skipped {email['sender_email']} (not in allowed list)")
                continue

            requested = extract_requested_filename(email["subject"])
            if not requested:
                send_auto_reply(
                    gmail, email["sender_email"], email["subject"],
                    "Please use subject format: 'Request: <file name>'. Example: Request: resume.pdf",
                )
                mark_as_read(gmail, email["msg_id"])
                results.append(f"HR: invalid subject from {email['sender_email']}")
                continue

            employee_id = get_employee_id_by_email(email["sender_email"])
            if not employee_id:
                send_auto_reply(
                    gmail, email["sender_email"], email["subject"],
                    "Your email was not found in our HR database. Please use your registered company email.",
                )
                mark_as_read(gmail, email["msg_id"])
                results.append(f"HR: unknown employee {email['sender_email']}")
                continue

            file_obj = fetch_employee_file(employee_id, requested)
            if not file_obj:
                suggestions = list_employee_files(employee_id, limit=10)
                suggestion_text = ", ".join(suggestions) if suggestions else "None"
                send_auto_reply(
                    gmail, email["sender_email"], email["subject"],
                    f"Could not find '{requested}' in your HR folder ({employee_id}).\n"
                    f"Available files: {suggestion_text}",
                )
                mark_as_read(gmail, email["msg_id"])
                results.append(f"HR: file '{requested}' not found for {email['sender_email']}")
                continue

            data, mime = file_obj
            send_reply_with_attachment(
                gmail, email["sender_email"], email["subject"],
                f"Hi, attaching '{requested}' as requested.", data, requested, mime,
            )
            mark_as_read(gmail, email["msg_id"])
            results.append(f"HR: sent '{requested}' to {email['sender_email']}")

        except Exception as e:
            results.append(f"HR: error for {email['sender_email']}: {e}")
            logger.exception(f"Error in HR node for {email['msg_id']}")

    return {"results": results}


def process_auto_reply_node(state: OrchestratorState) -> dict:
    config = state["config"]
    agent_conf = config.get("auto_mail_reply", {})
    results = list(state["results"])

    if not agent_conf.get("active", False):
        results.append("Auto Mail Reply agent is inactive — skipped")
        return {"results": results}

    auto_emails = state["classified"].get("auto_reply", [])
    if not auto_emails:
        return {"results": results}

    specific_mode = agent_conf.get("mode") == "specific"
    allowed = [e.lower() for e in agent_conf.get("emails", [])]

    gmail = gmail_service()

    for email in auto_emails:
        try:
            if specific_mode and email["sender_email"] not in allowed:
                mark_as_read(gmail, email["msg_id"])
                results.append(f"AutoReply: skipped {email['sender_email']} (not in target list)")
                continue

            reply_body = "Thank you for your email. Our team will respond shortly."
            send_auto_reply(gmail, email["sender_email"], email["subject"] or "(No Subject)", reply_body)
            mark_as_read(gmail, email["msg_id"])
            results.append(f"AutoReply: replied to {email['sender_email']}")

        except Exception as e:
            results.append(f"AutoReply: error for {email['sender_email']}: {e}")
            logger.exception(f"Error in auto-reply node for {email['msg_id']}")

    return {"results": results}


# ────────────────────────────────────────────────────────────────────
# Conditional routing
# ────────────────────────────────────────────────────────────────────

def should_continue(state: OrchestratorState) -> str:
    if not state.get("emails"):
        return "end"
    return "classify"


# ────────────────────────────────────────────────────────────────────
# Build the graph
# ────────────────────────────────────────────────────────────────────

workflow = StateGraph(OrchestratorState)

workflow.add_node("fetch_emails", fetch_emails_node)
workflow.add_node("classify_emails", classify_emails_node)
workflow.add_node("process_meetings", process_meetings_node)
workflow.add_node("process_hr_requests", process_hr_node)
workflow.add_node("process_auto_replies", process_auto_reply_node)

workflow.set_entry_point("fetch_emails")
workflow.add_conditional_edges("fetch_emails", should_continue, {
    "end": END,
    "classify": "classify_emails",
})
workflow.add_edge("classify_emails", "process_meetings")
workflow.add_edge("process_meetings", "process_hr_requests")
workflow.add_edge("process_hr_requests", "process_auto_replies")
workflow.add_edge("process_auto_replies", END)

graph = workflow.compile()


# ────────────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────────────

def run_orchestrator(config: dict | None = None) -> List[str]:
    if config is None:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
        else:
            config = {}

    initial_state: OrchestratorState = {
        "config": config,
        "emails": [],
        "classified": {"meeting": [], "hr": [], "auto_reply": []},
        "results": [],
    }

    final_state = graph.invoke(initial_state)
    for line in final_state["results"]:
        logger.info(line)
    return final_state["results"]


if __name__ == "__main__":
    results = run_orchestrator()
    for r in results:
        print(r)
