import streamlit as st
import json, os, time, threading

CONFIG_FILE = "agents_config.json"
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# --------------------------- Utility functions --------------------------- #
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            st.warning("⚠️ Config file corrupted. Resetting defaults.")
    return {
        "auto_mail_reply": {"active": False, "mode": "all", "emails": []},
        "meeting_scheduler": {"active": False, "mode": "all", "emails": []},
        "hr_document_request": {"active": False, "allowed_emails": ""},
    }

def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=4)

def run_orchestrator_background(config):
    """Run the LangGraph orchestrator in a background thread and write results to log."""
    from orchestrator import run_orchestrator
    try:
        results = run_orchestrator(config)
        log_file = os.path.join(LOG_DIR, "orchestrator.log")
        with open(log_file, "a", encoding="utf-8") as f:
            for line in results:
                f.write(line + "\n")
    except Exception as e:
        log_file = os.path.join(LOG_DIR, "orchestrator.log")
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"ERROR: {e}\n")

def run_agents(config):
    """Launch the LangGraph orchestrator to process all active agents."""
    thread = threading.Thread(target=run_orchestrator_background, args=(config,), daemon=True)
    thread.start()
    st.toast("🚀 LangGraph orchestrator started!", icon="✅")

def read_logs(agent_name):
    """Safely read log contents"""
    log_file = os.path.join(LOG_DIR, f"{agent_name}.log")
    if not os.path.exists(log_file):
        return "No logs yet for this agent."
    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()[-4000:]

# --------------------------- Sidebar navigation -------------------------- #
st.sidebar.title("AutoCorp Hub")
page = st.sidebar.radio(
    "Navigation",
    ["AI Automation Agents", "HR Agents", "Knowledge Base", "Dashboard"],
    label_visibility="collapsed"
)

config = load_config()

# --------------------------- Page 1: Configure Agents -------------------- #
if page == "AI Automation Agents":
    st.title("⚙️ Configure Agents")
    st.caption("Activate or configure individual AI automation agents.")

    # ---------- Auto Mail Reply Agent ----------
    st.subheader("📬 Auto Mail Reply Agent")
    active = st.toggle("Activate Agent",
                       value=config["auto_mail_reply"]["active"],
                       key="auto_reply_toggle")
    mode = st.radio(
        "Apply Auto Mail Reply To:",
        ("All Incoming Emails", "Specific Email Addresses"),
        index=0 if config["auto_mail_reply"]["mode"] == "all" else 1,
        key="auto_reply_mode"
    )
    if mode == "Specific Email Addresses":
        emails = st.text_area(
            "Enter email addresses (comma-separated)",
            value=", ".join(config["auto_mail_reply"]["emails"]),
            key="auto_reply_emails"
        )
        selected_emails = [e.strip() for e in emails.split(",") if e.strip()]
    else:
        selected_emails = []

    config["auto_mail_reply"].update({
        "active": active,
        "mode": "all" if mode == "All Incoming Emails" else "specific",
        "emails": selected_emails
    })

    st.divider()

    # ---------- Meeting Scheduler Agent ----------
    st.subheader("📅 Meeting Scheduler Agent")
    active_ms = st.toggle("Activate Agent",
                          value=config["meeting_scheduler"]["active"],
                          key="meeting_toggle")
    mode_ms = st.radio(
        "Apply Meeting Scheduler To:",
        ("All Incoming Emails", "Specific Email Addresses"),
        index=0 if config["meeting_scheduler"]["mode"] == "all" else 1,
        key="meeting_mode"
    )
    if mode_ms == "Specific Email Addresses":
        emails_ms = st.text_area(
            "Enter email addresses (comma-separated)",
            value=", ".join(config["meeting_scheduler"]["emails"]),
            key="meeting_emails"
        )
        selected_emails_ms = [e.strip() for e in emails_ms.split(",") if e.strip()]
    else:
        selected_emails_ms = []

    config["meeting_scheduler"].update({
        "active": active_ms,
        "mode": "all" if mode_ms == "All Incoming Emails" else "specific",
        "emails": selected_emails_ms
    })

    st.divider()
    if st.button("💾 Save & Run Active Agents"):
        save_config(config)
        st.success("Configuration saved successfully!")
        run_agents(config)
        st.info("✅ LangGraph orchestrator processing active agents.")

# --------------------------- Page 2: HR Agents --------------------------- #
elif page == "HR Agents":
    st.title("👩‍💼 HR Automation Agents")
    st.caption("Configure and monitor HR document request automation.")

    # ---------- HR Document Request Agent ----------
    st.subheader("📁 HR Document Request Agent")

    hr_config = config.get("hr_document_request", {"active": False, "allowed_emails": []})
    active_hr = st.toggle(
        "Activate Agent",
        value=hr_config.get("active", False),
        key="hr_doc_toggle"
    )

    emails_hr = st.text_area(
        "Enter allowed HR email addresses (comma-separated)",
        value=", ".join(hr_config.get("allowed_emails", [])),
        key="hr_doc_emails"
    )
    allowed_emails_hr = [e.strip() for e in emails_hr.split(",") if e.strip()]

    config["hr_document_request"] = {
        "active": active_hr,
        "allowed_emails": allowed_emails_hr
    }
    st.divider()
    if st.button("💾 Save & Run HR Agent"):
        save_config(config)
        st.success("Configuration saved successfully!")
        run_agents(config)
        st.info("✅ LangGraph orchestrator processing HR agent.")

    st.divider()
    st.subheader("🧾 HR Document Request Logs")
    log_text = read_logs("HR_Document_Request")
    st.text_area("Log Output", log_text, height=400)

# --------------------------- Page 3: Knowledge Base ---------------------- #
elif page == "Knowledge Base":
    from knowledge_base import render_knowledge_base_page
    render_knowledge_base_page()

# --------------------------- Page 4: Dashboard --------------------------- #
elif page == "Dashboard":
    st.title("📊 Active Agents Dashboard")
    st.caption("Monitor currently active agents and inspect their logs.")
    st.divider()

    # Orchestrator log
    with st.expander("🔗 LangGraph Orchestrator", expanded=True):
        orchestrator_log = read_logs("orchestrator")
        st.text_area("Orchestrator Log", orchestrator_log, height=200)

    st.divider()

    active_agents = []
    if config["auto_mail_reply"]["active"]:
        active_agents.append(("Auto Mail Reply", "mail"))
    if config["meeting_scheduler"]["active"]:
        active_agents.append(("Meeting Scheduler", "meeting_scheduler"))
    if config["hr_document_request"]["active"]:
        active_agents.append(("HR Document Request", "HR_Document_Request"))

    if not active_agents:
        st.info("No active agents at the moment.")
    else:
        for label, name in active_agents:
            with st.expander(f"🟢 {label}"):
                st.write(f"**Agent:** `{name}.py`")
                log_text = read_logs(name)
                log_box = st.empty()
                log_box.text(log_text)

                auto_refresh = st.checkbox(
                    f"Auto-refresh {label} logs",
                    value=False,
                    key=f"refresh_{name}"
                )
                if auto_refresh:
                    for _ in range(30):
                        log_box.text(read_logs(name))
                        time.sleep(5)
