"""Streamlit QA dashboard.

    streamlit run src/qa_agent/ui/Home.py

The UI is a client of the FastAPI backend only: it starts runs, approves
execution, submits review decisions and renders what the API returns. The
LangGraph workflow runs in the backend.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # src/, so `qa_agent` imports resolve

import streamlit as st  # noqa: E402

from qa_agent.logging_conf import configure_logging  # noqa: E402
from qa_agent.observability import context  # noqa: E402
from qa_agent.ui.api_client import BackendError  # noqa: E402
from qa_agent.ui.components import (  # noqa: E402
    OUTCOME,
    RUN_STATUS,
    active_project,
    active_run_id,
    badge,
    get_client,
    inject_css,
    session_id,
    set_active_project,
    set_active_run,
)
from qa_agent.ui.views import analysis, design, execution, report, setup  # noqa: E402

configure_logging()
st.set_page_config(page_title="AI QA Agent", page_icon="🧪", layout="wide")
inject_css()

PAGES = {
    "projects": st.Page(setup.project_selection, title="Project Selection", icon="📁", url_path="projects", default=True),
    "requirement": st.Page(setup.requirement_input, title="Requirement Input", icon="📝", url_path="requirement"),
    "generate": st.Page(setup.generate_test_cases, title="Generate Test Cases", icon="✨", url_path="generate"),
    "test_cases": st.Page(design.test_cases, title="Test Cases", icon="📋", url_path="test-cases"),
    "test_plan": st.Page(design.test_plan, title="Test Plan", icon="🗺️", url_path="test-plan"),
    "execute": st.Page(execution.execute_tests, title="Execute Tests", icon="▶️", url_path="execute"),
    "status": st.Page(execution.run_status, title="Test Run Status", icon="📡", url_path="status"),
    "results": st.Page(execution.test_results, title="Test Results", icon="📊", url_path="results"),
    "failures": st.Page(execution.failure_details, title="Failure Details", icon="🔍", url_path="failures"),
    "root_cause": st.Page(analysis.root_cause, title="Root Cause Analysis", icon="🧠", url_path="root-cause"),
    "bugs": st.Page(analysis.bug_reports, title="Bug Report", icon="🐞", url_path="bugs"),
    "confidence": st.Page(analysis.confidence, title="Confidence Score", icon="🎯", url_path="confidence"),
    "review": st.Page(analysis.human_review, title="Human Review", icon="🧑‍⚖️", url_path="review"),
    "report": st.Page(report.final_report, title="Final QA Report", icon="📄", url_path="report"),
}
st.session_state["_pages"] = PAGES

navigation = st.navigation(
    {
        "Setup": [PAGES["projects"], PAGES["requirement"], PAGES["generate"]],
        "Test design": [PAGES["test_cases"], PAGES["test_plan"]],
        "Execution": [PAGES["execute"], PAGES["status"], PAGES["results"], PAGES["failures"]],
        "Analysis": [PAGES["root_cause"], PAGES["bugs"], PAGES["confidence"], PAGES["review"]],
        "Report": [PAGES["report"]],
    },
    expanded=True,
)


@st.cache_data(ttl=5, show_spinner=False)
def _backend_status() -> dict:
    client = get_client()
    client.health()
    return client.system_status()


def _run_label(run: dict) -> str:
    status = RUN_STATUS.get(run["status"], (run["status"], "•", ""))
    outcome = f" → {OUTCOME[run['outcome']][0]}" if run.get("outcome") in OUTCOME else ""
    text = run["requirement"].replace("\n", " ")
    return f"{status[1]} {run['run_id'][:8]} · {status[0]}{outcome} · {text[:40]}"


with st.sidebar:
    st.markdown("## 🧪 AI QA Agent")
    try:
        system = _backend_status()
    except BackendError as exc:
        system = None
        st.markdown(badge("error", label="Backend offline"), unsafe_allow_html=True)
        st.caption(exc.message)
        if exc.hint:
            st.caption(exc.hint)
        if st.button("Retry connection", width="stretch"):
            _backend_status.clear()
            st.rerun()

    if system is not None:
        st.markdown(badge("completed", label="Backend connected"), unsafe_allow_html=True)
        if system["warnings"]:
            with st.expander(f"⚠️ {len(system['warnings'])} configuration notes"):
                for warning in system["warnings"]:
                    st.caption(warning)

        try:
            client = get_client()
            projects = [p["name"] for p in client.list_projects()]
            if active_project() not in projects:
                projects.append(active_project())
            # Pages also change the active project/run, so the widgets mirror session state
            # (set before they render) and write back only when the user changes them.
            st.session_state["_project_select"] = active_project()
            st.selectbox(
                "Project",
                projects,
                key="_project_select",
                on_change=lambda: (set_active_project(st.session_state["_project_select"]), set_active_run(None)),
            )

            runs = client.list_runs(active_project())
            ids = [r["run_id"] for r in runs]
            if active_run_id() and active_run_id() not in ids:
                set_active_run(None)
            if runs:
                labels = {r["run_id"]: _run_label(r) for r in runs}
                st.session_state["_run_select"] = active_run_id()
                st.selectbox(
                    "Active run",
                    ids,
                    key="_run_select",
                    format_func=labels.get,
                    placeholder="Select a run",
                    on_change=lambda: set_active_run(st.session_state["_run_select"]),
                )
            else:
                st.caption("No runs in this project yet.")
        except BackendError as exc:
            st.caption(f"Could not load projects/runs: {exc.message}")

        if st.button("＋ New run", width="stretch", type="primary"):
            st.switch_page(PAGES["requirement"])

if system is None:
    st.title("Backend unavailable")
    st.error(
        "The dashboard cannot reach the QA Agent API, so no data can be shown. "
        "Nothing is lost: runs live in the backend process.",
        icon="🔌",
    )
    st.code(
        "uvicorn qa_agent.api.main:app --app-dir src --port 8000\n"
        f"# the dashboard expects it at {get_client().base_url} (STREAMLIT_API_BASE_URL)",
        language="bash",
    )
    st.stop()

with context.bind(client_session=session_id(), run_id=active_run_id(), project=active_project()):
    navigation.run()
