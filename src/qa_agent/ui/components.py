"""Shared UI pieces: API access with error handling, status badges, pipeline tracker,
session state for the active project/run, and live refresh while a run works.

Presentation only: statuses and data come from the backend as-is.
"""

import html
from collections.abc import Callable
from datetime import datetime
from typing import Any, TypeVar
from uuid import uuid4

import streamlit as st

from qa_agent.config import get_settings
from qa_agent.ui.api_client import BackendError, QAApiClient

T = TypeVar("T")

POLL_INTERVAL_S = 2

# --- status vocabulary ---------------------------------------------------------
# Tones use the reserved status palette; a badge always carries icon + label, never color alone.
TONES = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
    "info": "#3b82c4",
    "neutral": "#8b8d98",
}

RUN_STATUS = {
    "running": ("Running", "◌", "info"),
    "awaiting_execution": ("Waiting for Approval", "⏸", "warning"),
    "awaiting_review": ("Waiting for Approval", "⏸", "warning"),
    "completed": ("Completed", "✓", "good"),
    "error": ("Error", "✕", "critical"),
}
OUTCOME = {
    "passed": ("Passed", "✓", "good"),
    "failed": ("Failed", "✕", "critical"),
    "rejected": ("Rejected", "⊘", "serious"),
    "error": ("Error", "✕", "critical"),
    "inconclusive": ("Inconclusive", "?", "warning"),
}
TEST_STATUS = {
    "passed": ("Passed", "✓", "good"),
    "failed": ("Failed", "✕", "critical"),
    "error": ("Error", "!", "serious"),
    "skipped": ("Skipped", "–", "neutral"),
}
NODE_STATUS = {
    "pending": ("Pending", "○", "neutral"),
    "running": ("Running", "◌", "info"),
    "waiting": ("Waiting", "⏸", "warning"),
    "completed": ("Completed", "✓", "good"),
    "error": ("Error", "✕", "critical"),
    "skipped": ("Skipped", "–", "neutral"),
}
SEVERITY = {
    "critical": ("Critical", "▲", "critical"),
    "high": ("High", "▲", "serious"),
    "medium": ("Medium", "■", "warning"),
    "low": ("Low", "▼", "neutral"),
}

NODE_LABELS = {
    "requirement_analyzer": "Analyze requirement",
    "test_case_generator": "Generate test cases",
    "test_planner": "Plan tests",
    "test_executor": "Execute tests",
    "result_analyzer": "Analyze results",
    "failure_analyzer": "Analyze failures",
    "root_cause_analyzer": "Root cause analysis",
    "bug_report_generator": "Write bug reports",
    "confidence_checker": "Check confidence",
    "human_review": "Human review",
    "final_report_generator": "Final QA report",
}
STAGES = [
    ("Design", ["requirement_analyzer", "test_case_generator", "test_planner"]),
    ("Execution", ["test_executor", "result_analyzer"]),
    ("Diagnosis", ["failure_analyzer", "root_cause_analyzer", "bug_report_generator"]),
    ("Decision", ["confidence_checker", "human_review", "final_report_generator"]),
]

CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.qa-badge {display:inline-flex; align-items:center; gap:.35rem; padding:.12rem .6rem; border-radius:999px;
  font-size:.8rem; font-weight:600; line-height:1.4; white-space:nowrap; border:1px solid var(--tone);
  background: color-mix(in srgb, var(--tone) 14%, transparent);}
.qa-badge .ic {color: var(--tone); font-weight:700;}
.qa-muted {opacity:.68; font-size:.85rem;}
.qa-header {display:flex; flex-wrap:wrap; align-items:center; gap:.6rem; margin:.1rem 0 .6rem 0;}
.qa-header .rid {font-family: ui-monospace, monospace; font-size:.85rem; opacity:.7;}
.qa-pipeline {display:grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap:.75rem; margin:.4rem 0 1rem 0;}
@media (max-width: 900px) {.qa-pipeline {grid-template-columns: repeat(2, minmax(0,1fr));}}
.qa-stage {border:1px solid rgba(128,128,128,.25); border-radius:.6rem; padding:.55rem .65rem;}
.qa-stage h4 {margin:0 0 .4rem 0; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; opacity:.65;}
.qa-step {display:flex; align-items:center; gap:.5rem; padding:.3rem .45rem; margin:.2rem 0; border-radius:.4rem;
  border-left:3px solid var(--tone); background: color-mix(in srgb, var(--tone) 8%, transparent); font-size:.86rem;}
.qa-step .ic {color: var(--tone); font-weight:700; width:1rem; text-align:center;}
.qa-step .meta {margin-left:auto; font-size:.72rem; opacity:.7; white-space:nowrap;}
.qa-step.running .ic {animation: qa-pulse 1.2s ease-in-out infinite;}
@keyframes qa-pulse {0%,100% {opacity:1} 50% {opacity:.25}}
.qa-card {border:1px solid rgba(128,128,128,.25); border-radius:.6rem; padding:.8rem 1rem; margin-bottom:.6rem;}
.qa-kv {display:grid; grid-template-columns: max-content 1fr; gap:.2rem .9rem; font-size:.9rem;}
.qa-kv dt {opacity:.65;}
.qa-kv dd {margin:0;}
.qa-meter {height:.7rem; border-radius:999px; background: rgba(128,128,128,.18); position:relative; margin:.4rem 0 1.4rem 0;}
.qa-meter .fill {height:100%; border-radius:999px; background: var(--tone);}
.qa-meter .mark {position:absolute; top:-.3rem; width:2px; height:1.3rem; background: currentColor; opacity:.8;}
.qa-meter .mark span {position:absolute; top:1.35rem; transform:translateX(-50%); font-size:.72rem; white-space:nowrap; opacity:.85;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def badge(status: str | None, mapping: dict[str, tuple[str, str, str]] = RUN_STATUS, *, label: str | None = None) -> str:
    text, icon, tone = mapping.get(status or "", (str(status or "Unknown").replace("_", " ").title(), "•", "neutral"))
    return (
        f'<span class="qa-badge" style="--tone:{TONES[tone]}"><span class="ic">{icon}</span>'
        f"{html.escape(label or text)}</span>"
    )


def show_badges(*badges: str) -> None:
    st.markdown(" ".join(badges), unsafe_allow_html=True)


def esc(value: Any) -> str:
    return html.escape(str(value))


def fmt_time(value: str | None) -> str:
    if not value:
        return "—"
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%H:%M:%S")
    except ValueError:
        return value


def fmt_ms(ms: int | float | None) -> str:
    if ms is None:
        return "—"
    return f"{ms:.0f} ms" if ms < 1000 else f"{ms / 1000:.1f} s"


def pct(value: float | None) -> str:
    return "—" if value is None else f"{value:.0%}"


# --- backend access ------------------------------------------------------------


def session_id() -> str | None:
    """Stable id of this browser session, sent to the backend as X-Client-Session."""
    try:
        if "_client_session" not in st.session_state:
            st.session_state["_client_session"] = uuid4().hex
        return st.session_state["_client_session"]
    except Exception:  # no Streamlit session (e.g. imported outside `streamlit run`)
        return None


@st.cache_resource
def get_client() -> QAApiClient:
    return QAApiClient(get_settings().streamlit_api_base_url, session_id=session_id)


def show_backend_error(exc: BackendError) -> None:
    if exc.unreachable:
        st.error(f"**Backend unavailable.** {exc.message}", icon="🔌")
    elif exc.status_code and exc.status_code >= 500:
        st.error(f"**Backend error ({exc.status_code}).** {exc.message}", icon="🛑")
    else:
        st.warning(f"**Request not accepted ({exc.status_code}).** {exc.message}", icon="⚠️")
    if exc.hint:
        st.caption(exc.hint)
    if exc.request_id:
        st.caption(f"Reference: request `{exc.request_id[:8]}` (search the backend log for `req={exc.request_id[:8]}`)")


def call(fn: Callable[..., T], *args: Any, **kwargs: Any) -> T | None:
    """Call the backend; on failure show the error and return None."""
    try:
        return fn(*args, **kwargs)
    except BackendError as exc:
        show_backend_error(exc)
        return None


# --- session: active project and run ------------------------------------------


def active_project() -> str:
    return st.session_state.get("project", "default")


def set_active_project(name: str) -> None:
    st.session_state["project"] = name


def active_run_id() -> str | None:
    return st.session_state.get("run_id")


def set_active_run(run_id: str | None) -> None:
    st.session_state["run_id"] = run_id


def fetch_run(run_id: str) -> dict[str, Any] | None:
    try:
        return get_client().get_run(run_id)
    except BackendError as exc:
        if exc.status_code == 404:
            set_active_run(None)
            st.warning("This run is no longer known to the backend (it keeps runs in memory and may have restarted).")
        else:
            show_backend_error(exc)
        return None


# --- page scaffolding ----------------------------------------------------------


def page_title(title: str, caption: str | None = None) -> None:
    st.title(title)
    if caption:
        st.caption(caption)


def run_header(run: dict[str, Any]) -> None:
    parts = [badge(run["status"])]
    if run.get("outcome"):
        parts.append(badge(run["outcome"], OUTCOME, label=f"Outcome: {OUTCOME.get(run['outcome'], (run['outcome'],))[0]}"))
    if run["status"] == "awaiting_execution":
        detail = "Approve to execute the planned tests"
    elif run["status"] == "awaiting_review":
        detail = "A reviewer must approve the analysis"
    elif run["status"] == "running":
        current = ", ".join(NODE_LABELS.get(n, n) for n in run.get("current_nodes") or []) or "starting"
        detail = f"Now: {current}"
    else:
        detail = ""
    requirement = run["requirement"]
    short = requirement if len(requirement) <= 110 else requirement[:107] + "…"
    st.markdown(
        f'<div class="qa-header">{" ".join(parts)}<span class="qa-muted">{esc(detail)}</span>'
        f'<span class="rid">run {esc(run["run_id"][:8])} · {esc(run["project"])}</span></div>'
        f'<div class="qa-muted" style="margin-bottom:.8rem">“{esc(short)}”</div>',
        unsafe_allow_html=True,
    )
    if run["status"] == "error" and run.get("error"):
        st.error(f"The run stopped unexpectedly: {run['error']}", icon="🛑")
    ticket_verdict_banner(run)


VERDICT_STYLE = {
    "confirmed": (st.error, "🐞"),
    "not_reproduced": (st.success, "✅"),
    "inconclusive": (st.warning, "⚠️"),
    "pending": (st.info, "⏳"),
}


def ticket_verdict_banner(run: dict[str, Any]) -> None:
    """Bug ticket runs: the verdict (confirmed / not reproducible / inconclusive) on every run page."""
    verdict = run.get("verdict")
    if not verdict:
        return
    ticket = (run.get("metadata") or {}).get("bug_ticket") or {}
    show, icon = VERDICT_STYLE.get(verdict["status"], (st.info, "ℹ️"))
    label = f"{ticket.get('ticket_id')} · " if ticket.get("ticket_id") else ""
    show(f"**{label}{verdict['headline']}**  \n{verdict['explanation']}", icon=icon)


def pipeline(run: dict[str, Any]) -> None:
    nodes = {n["name"]: n for n in run.get("nodes") or []}
    stages = []
    for stage, names in STAGES:
        steps = []
        for name in names:
            node = nodes.get(name, {"status": "pending"})
            text, icon, tone = NODE_STATUS[node["status"]]
            meta = text if node["status"] in ("running", "waiting", "error") else ""
            if node["status"] == "completed":
                meta = fmt_ms(node.get("duration_ms")) + (f" · ×{node['visits']}" if node.get("visits", 0) > 1 else "")
            steps.append(
                f'<div class="qa-step {node["status"]}" style="--tone:{TONES[tone]}" title="{esc(text)}">'
                f'<span class="ic">{icon}</span>{esc(NODE_LABELS[name])}<span class="meta">{esc(meta)}</span></div>'
            )
        stages.append(f'<div class="qa-stage"><h4>{stage}</h4>{"".join(steps)}</div>')
    st.markdown(f'<div class="qa-pipeline">{"".join(stages)}</div>', unsafe_allow_html=True)


def empty_state(message: str, *, icon: str = "ℹ️") -> None:
    st.info(message, icon=icon)


def require_run() -> str:
    """The active run id, or stop the page with a pointer to start one."""
    run_id = active_run_id()
    if not run_id:
        empty_state("No run selected. Pick one in the sidebar, or start a new one from **Requirement Input**.", icon="🧭")
        if st.button("Go to Requirement Input", type="primary"):
            st.switch_page(st.session_state["_pages"]["requirement"])
        st.stop()
    return run_id


def next_step(label: str, page_key: str) -> None:
    st.divider()
    if st.button(f"{label} →", key=f"next-{page_key}"):
        st.switch_page(st.session_state["_pages"][page_key])


def run_page(body: Callable[[dict[str, Any]], None]) -> None:
    """Render a page for the active run; re-render live every few seconds while it runs.

    When the run leaves ``running`` the whole app reruns, so the sidebar and
    action buttons pick up the new status too.
    """
    run_id = require_run()
    run = fetch_run(run_id)
    if run is None:
        st.stop()

    def render(current: dict[str, Any]) -> None:
        run_header(current)
        body(current)

    if run["status"] != "running":
        render(run)
        return

    @st.fragment(run_every=POLL_INTERVAL_S)
    def live() -> None:
        current = fetch_run(run_id)
        if current is None:
            return
        render(current)
        if current["status"] != "running":
            st.rerun()

    live()
