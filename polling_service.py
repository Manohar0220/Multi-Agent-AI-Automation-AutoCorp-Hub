"""Process-wide email polling service for the Streamlit application."""

import json
import logging
import os
import threading
from datetime import datetime, timedelta
from typing import Any, Dict


CONFIG_FILE = "agents_config.json"
POLL_INTERVAL_SECONDS = int(os.getenv("EMAIL_POLL_INTERVAL_SECONDS", "300"))
LOG_DIR = "logs"
LOG_FILE = os.path.join(LOG_DIR, "polling_service.log")

os.makedirs(LOG_DIR, exist_ok=True)

logger = logging.getLogger("email_polling_service")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)


_control_lock = threading.Lock()
_status_lock = threading.Lock()
_stop_event = threading.Event()
_wake_event = threading.Event()
_polling_thread = None

_status = {
    "running": False,
    "stop_requested": False,
    "last_started_at": None,
    "last_completed_at": None,
    "next_check_at": None,
    "last_error": None,
    "last_result_count": 0,
}


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _set_status(**updates):
    with _status_lock:
        _status.update(updates)


def _has_active_agents(config: Dict[str, Any]) -> bool:
    return any(
        config.get(agent_name, {}).get("active", False)
        for agent_name in (
            "auto_mail_reply",
            "meeting_scheduler",
            "hr_document_request",
        )
    )


def _load_config() -> Dict[str, Any]:
    with open(CONFIG_FILE, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def _run_orchestrator_once(config: Dict[str, Any]):
    # Import lazily so opening the UI does not initialize every Google service
    # until polling is actually enabled.
    from orchestrator import run_orchestrator

    return run_orchestrator(config)


def _polling_loop():
    global _polling_thread

    logger.info("Email polling started (interval=%s seconds)", POLL_INTERVAL_SECONDS)
    _set_status(running=True, stop_requested=False, last_error=None)

    try:
        while not _stop_event.is_set():
            try:
                config = _load_config()
            except Exception as exc:
                logger.exception("Unable to load agent configuration")
                _set_status(last_error=f"Configuration error: {exc}")
                config = {}

            if not _has_active_agents(config):
                logger.info("No active agents remain; email polling is stopping")
                break

            _set_status(
                last_started_at=_now(),
                next_check_at=None,
                last_error=None,
            )

            try:
                results = _run_orchestrator_once(config)
                _set_status(
                    last_completed_at=_now(),
                    last_result_count=len(results),
                )
                logger.info("Polling cycle completed with %s result messages", len(results))
            except Exception as exc:
                logger.exception("Email polling cycle failed")
                _set_status(last_completed_at=_now(), last_error=str(exc))

            if _stop_event.is_set():
                break

            next_check = datetime.now().astimezone() + timedelta(seconds=POLL_INTERVAL_SECONDS)
            _set_status(next_check_at=next_check.isoformat(timespec="seconds"))

            # A saved configuration update wakes the service immediately.
            # Otherwise, the next mailbox check occurs after the interval.
            _wake_event.wait(POLL_INTERVAL_SECONDS)
            _wake_event.clear()
    finally:
        logger.info("Email polling stopped")
        _set_status(running=False, stop_requested=False, next_check_at=None)
        with _control_lock:
            _polling_thread = None


def configure_email_polling(config: Dict[str, Any]) -> str:
    """Start, refresh, or stop polling to match the saved configuration."""
    global _polling_thread

    if not _has_active_agents(config):
        stop_email_polling()
        return "stopped"

    with _control_lock:
        if _polling_thread and _polling_thread.is_alive():
            # Cancel a pending stop if an agent was quickly reactivated, then
            # wake the worker so it reloads the latest saved configuration.
            _stop_event.clear()
            _wake_event.set()
            logger.info("Agent configuration updated; polling service was awakened")
            return "updated"

        _stop_event.clear()
        _wake_event.clear()
        _polling_thread = threading.Thread(
            target=_polling_loop,
            name="autocorp-email-poller",
            daemon=True,
        )
        _polling_thread.start()
        return "started"


def stop_email_polling():
    """Request a graceful stop after the current polling cycle completes."""
    with _control_lock:
        is_running = bool(_polling_thread and _polling_thread.is_alive())
        if is_running:
            _set_status(stop_requested=True, next_check_at=None)
            _stop_event.set()
            _wake_event.set()
            logger.info("Email polling stop requested")
    return is_running


def get_email_polling_status() -> Dict[str, Any]:
    with _control_lock:
        is_running = bool(
            _polling_thread and _polling_thread.is_alive()
        )
    with _status_lock:
        current_status = dict(_status)
    current_status["running"] = is_running
    current_status["interval_seconds"] = POLL_INTERVAL_SECONDS
    return current_status
