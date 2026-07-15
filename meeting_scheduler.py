import os
import re
import datetime as dt
import json
from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from mail import gmail_service, get_unread_emails, read_email, send_auto_reply
import sys
from google.auth.transport.requests import Request
import datetime

load_dotenv()

CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar']
CONFIG_FILE = "agents_config.json"


LOG_FILE = "logs/meeting_scheduler.log"
os.makedirs("logs", exist_ok=True)

# ------------------------------------------------------------------
# Calendar Authentication
# ------------------------------------------------------------------
def google_calendar_service():
    creds = None
    token_file = os.getenv("CALENDAR_TOKEN_FILE")
    creds_file = os.getenv("CALENDAR_CREDENTIALS_FILE")

    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, CALENDAR_SCOPES)

        # Auto-refresh logic
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        elif creds and creds.expiry and creds.expiry < datetime.datetime.utcnow() + datetime.timedelta(minutes=5):
            creds.refresh(Request())

        # Save refreshed creds
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    else:
        flow = InstalledAppFlow.from_client_secrets_file(creds_file, CALENDAR_SCOPES)
        creds = flow.run_local_server(port=8081)
        with open(token_file, "w") as token:
            token.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)

# ------------------------------------------------------------------
# Utility: Parse meeting datetime from email text
# ------------------------------------------------------------------
def parse_meeting_datetime(body):
    txt = re.sub(r'(\d{1,2})(st|nd|rd|th)\b', r'\1', body.strip(), flags=re.IGNORECASE)
    date_day_month = re.search(
        r'\b(\d{1,2})\s+(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'[, ]+\s*(\d{4})\b', txt, flags=re.IGNORECASE)
    date_month_day = re.search(
        r'\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|'
        r'Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)'
        r'\s+(\d{1,2}),?\s+(\d{4})\b', txt, flags=re.IGNORECASE)

    if date_day_month:
        d, mon, y = date_day_month.groups()
        date_str, date_fmts = f"{d} {mon} {y}", ["%d %b %Y", "%d %B %Y"]
    elif date_month_day:
        mon, d, y = date_month_day.groups()
        date_str, date_fmts = f"{mon} {d}, {y}", ["%b %d, %Y", "%B %d, %Y"]
    else:
        return None

    time_regex = r'(\d{1,2}(?::\d{2})?\s*(?:am|pm))'
    times = re.findall(time_regex, txt, flags=re.IGNORECASE)
    start_str, end_str = (times[0] if times else None), (times[1] if len(times) > 1 else None)

    def try_parse(dt_date, t):
        from datetime import datetime
        for tf in ("%I%p", "%I:%M%p"):
            for df in date_fmts:
                try:
                    return datetime.strptime(f"{dt_date} {t}", f"{df} {tf}")
                except ValueError:
                    pass
        return None

    if not start_str:
        at_match = re.search(r'\bat\s+' + time_regex, txt, flags=re.IGNORECASE)
        if at_match:
            start_str = at_match.group(1)
    if not start_str:
        return None

    start_dt = try_parse(date_str, start_str)
    if not start_dt:
        return None

    end_dt = try_parse(date_str, end_str) if end_str else start_dt + dt.timedelta(hours=1)
    if end_dt <= start_dt:
        end_dt = start_dt + dt.timedelta(hours=1)

    return start_dt, end_dt


# ------------------------------------------------------------------
# Calendar Operations
# ------------------------------------------------------------------
def check_calendar_conflict(service, start_time, end_time):
    events = service.events().list(
        calendarId='primary',
        timeMin=start_time.isoformat() + 'Z',
        timeMax=end_time.isoformat() + 'Z',
        singleEvents=True,
        orderBy='startTime'
    ).execute().get('items', [])
    return len(events) > 0


def schedule_calendar_event(service, summary, start_time, end_time):
    event = {
        'summary': summary,
        'start': {'dateTime': start_time.isoformat(), 'timeZone': 'America/Denver'},
        'end': {'dateTime': end_time.isoformat(), 'timeZone': 'America/Denver'}
    }
    return service.events().insert(calendarId='primary', body=event, sendUpdates='all').execute()


# ------------------------------------------------------------------
# Main Agent Logic
# ------------------------------------------------------------------
def main():
    if not os.path.exists(CONFIG_FILE):
        print(" Config file not found. Run Streamlit dashboard first.")
        return
    config = json.load(open(CONFIG_FILE))

    agent_conf = config.get("meeting_scheduler", {})
    if not agent_conf.get("active", False):
        print(" Meeting Scheduler Agent is inactive.")
        return

    specific_mode = agent_conf.get("mode") == "specific"
    allowed_emails = [e.lower() for e in agent_conf.get("emails", [])]

    gmail = gmail_service()
    calendar = google_calendar_service()

    messages = get_unread_emails(gmail)
    if not messages:
        print(" No new emails found.")
        return

    for msg in messages:
        msg_id = msg['id']
        subject, sender, body = read_email(gmail, msg_id)
        if not subject or not sender:
            continue

        print(f"\n Checking email from {sender} — Subject: {subject}")
        sender_email = sender.split("<")[-1].replace(">", "").strip().lower()

        if "schedule a meet" not in subject.lower():
            continue
        if specific_mode and sender_email not in allowed_emails:
            print(f" Skipping {sender_email} (not in target list)")
            continue

        print(" Meeting request detected! Parsing details...")
        meeting_times = parse_meeting_datetime(body)
        if not meeting_times:
            print(" Could not parse meeting date/time from email body.")
            continue

        meeting_start, meeting_end = meeting_times
        conflict = check_calendar_conflict(calendar, meeting_start, meeting_end)

        if conflict:
            reply = f"Hi, I already have an event at {meeting_start.strftime('%I:%M %p, %b %d %Y')}."
        else:
            event = schedule_calendar_event(calendar, "Auto-Scheduled Meeting", meeting_start, meeting_end)
            reply = f"Your meeting has been scheduled for {meeting_start.strftime('%I:%M %p, %b %d %Y')}.\n\nEvent: {event.get('htmlLink')}"

        send_auto_reply(gmail, sender_email, subject, reply)


if __name__ == "__main__":
    sys.stdout = open(LOG_FILE, "w", encoding="utf-8", buffering=1)
    sys.stderr = sys.stdout
    main()
