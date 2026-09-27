"""Screens 1-3: project selection, requirement input, test case generation."""

import streamlit as st

from qa_agent.ui.components import (
    NODE_LABELS,
    OUTCOME,
    active_project,
    badge,
    call,
    empty_state,
    get_client,
    next_step,
    page_title,
    pipeline,
    run_page,
    set_active_project,
    set_active_run,
)

SAMPLE_REQUIREMENTS = {
    "Password reset (REQ-AUTH-003)": (
        "A user who forgot their password can reset it using their registered email address.\n\n"
        "Acceptance criteria:\n"
        "1. The user enters their email on the \"Forgot password\" page and submits it.\n"
        "2. If the email is registered, a reset link is emailed within 2 minutes.\n"
        "3. The page shows the same confirmation message whether or not the email is registered.\n"
        "4. The reset link expires after 30 minutes and can be used only once.\n"
        "5. The new password must satisfy the password policy.\n"
        "6. After a successful reset, existing sessions are signed out and the user can log in with the new password.\n"
        "7. No more than 5 reset requests per email per hour are accepted."
    ),
    "Password reset (one line)": "User should be able to reset password using registered email.",
    "Login": (
        "A registered user can log in with email and password. Invalid credentials show a generic error "
        "message without revealing whether the email exists. After 5 failed attempts within 15 minutes "
        "the account is temporarily locked."
    ),
}

DESIGN_NODES = ("requirement_analyzer", "test_case_generator", "test_planner")


def project_selection() -> None:
    page_title("Project Selection", "Runs are grouped by project. Pick the project to work in.")
    client = get_client()
    projects = call(client.list_projects)
    if projects is None:
        st.stop()

    current = active_project()
    cols = st.columns(3)
    for i, project in enumerate(projects):
        with cols[i % 3], st.container(border=True):
            st.markdown(f"**{project['name']}**" + ("  ·  _active_" if project["name"] == current else ""))
            st.caption(project.get("description") or "No description")
            st.metric("Runs", project["run_count"], label_visibility="visible")
            if st.button(
                "Selected" if project["name"] == current else "Select",
                key=f"select-{project['name']}",
                disabled=project["name"] == current,
                width="stretch",
            ):
                set_active_project(project["name"])
                set_active_run(None)
                st.rerun()

    with st.expander("Create a project", expanded=len(projects) <= 1):
        with st.form("create-project", clear_on_submit=True):
            name = st.text_input("Project name", placeholder="e.g. Auth Service")
            description = st.text_input("Description (optional)")
            if st.form_submit_button("Create project", type="primary"):
                if not name.strip():
                    st.warning("Enter a project name.")
                elif (created := call(client.create_project, name.strip(), description.strip())) is not None:
                    set_active_project(created["name"])
                    set_active_run(None)
                    st.success(f"Project **{created['name']}** created and selected.")
                    st.rerun()

    st.subheader(f"Runs in {current}")
    runs = call(client.list_runs, current)
    if runs is None:
        return
    if not runs:
        empty_state("No runs in this project yet. Start one from **Requirement Input**.")
    for run in runs:
        with st.container(border=True):
            left, right = st.columns([5, 1])
            with left:
                parts = [badge(run["status"])]
                if run.get("outcome"):
                    parts.append(badge(run["outcome"], OUTCOME))
                st.markdown(" ".join(parts) + f' <span class="qa-muted">run {run["run_id"][:8]}</span>', unsafe_allow_html=True)
                st.caption(run["requirement"][:160] + ("…" if len(run["requirement"]) > 160 else ""))
            with right:
                if st.button("Open", key=f"open-{run['run_id']}", width="stretch"):
                    set_active_run(run["run_id"])
                    st.switch_page(st.session_state["_pages"]["status"])
    next_step("Next: Requirement Input", "requirement")


def requirement_input() -> None:
    page_title("Requirement Input", f"Describe what to test. The run is created in project **{active_project()}**.")

    mode = st.radio(
        "What do you want to test?",
        ["🐞 Bug ticket from the service desk", "📄 Requirement"],
        horizontal=True,
        key="input_mode",
        help="A bug ticket is verified: the agent follows its steps to reproduce and reports whether "
        "the bug is confirmed or not reproducible.",
    )
    if mode.startswith("🐞"):
        _bug_ticket_form()
        return

    sample = st.selectbox("Start from a sample (optional)", ["—", *SAMPLE_REQUIREMENTS], key="sample")
    if sample != "—" and st.session_state.get("_last_sample") != sample:
        st.session_state["requirement_text"] = SAMPLE_REQUIREMENTS[sample]
        st.session_state["_last_sample"] = sample

    with st.form("requirement"):
        text = st.text_area(
            "Requirement",
            key="requirement_text",
            height=220,
            placeholder="e.g. User should be able to reset password using registered email.",
        )
        c1, c2, c3 = st.columns(3)
        ticket = c1.text_input("Ticket / requirement id", placeholder="REQ-AUTH-003")
        component = c2.text_input("Component", placeholder="auth")
        priority = c3.selectbox("Business priority", ["", "high", "medium", "low"])
        gate = st.toggle(
            "Pause for approval before executing tests",
            value=True,
            help="The agent stops after planning so you can review the test cases and plan, "
            "then start execution from **Execute Tests**.",
        )
        submitted = st.form_submit_button("Generate Test Cases", type="primary", width="stretch")

    if submitted:
        if not text.strip():
            st.warning("Enter a requirement first.")
            return
        metadata = {k: v for k, v in {"ticket": ticket, "component": component, "priority": priority}.items() if v}
        run = call(
            get_client().start_run, text.strip(), project=active_project(), metadata=metadata, pause_before_execution=gate
        )
        if run is not None:
            set_active_run(run["run_id"])
            st.switch_page(st.session_state["_pages"]["generate"])


def _bug_ticket_form() -> None:
    st.caption(
        "Copy the ticket from the service desk. The agent turns it into reproduction tests, runs them "
        "against the configured test environment and gives a verdict: **bug confirmed**, "
        "**not reproducible** or **inconclusive**, with the test evidence."
    )
    with st.form("bug_ticket"):
        c1, c2 = st.columns([1, 3])
        ticket_id = c1.text_input("Ticket number", placeholder="SD-1042")
        title = c2.text_input("Title *", placeholder="Account is not locked after failed logins")
        description = st.text_area(
            "Description", height=90, placeholder="What the customer reported, in their words."
        )
        steps = st.text_area(
            "Steps to reproduce (one per line)",
            height=120,
            placeholder="Open the login page\nEnter a wrong password 3 times\nTry to log in again",
        )
        c1, c2 = st.columns(2)
        expected = c1.text_area("Expected result *", height=90, placeholder="The account is locked")
        actual = c2.text_area("Actual result (reported) *", height=90, placeholder="The login still says invalid password")
        c1, c2 = st.columns([3, 1])
        environment = c1.text_input("Environment / URL", placeholder="http://127.0.0.1:8765/login.html")
        priority = c2.selectbox("Priority", ["", "high", "medium", "low"])
        gate = st.toggle(
            "Pause for approval before executing tests",
            value=False,
            help="Review the generated reproduction tests before they run.",
        )
        submitted = st.form_submit_button("Verify bug", type="primary", width="stretch")

    if not submitted:
        return
    missing = [name for name, value in (("Title", title), ("Expected result", expected), ("Actual result", actual)) if not value.strip()]
    if missing:
        st.warning(f"Fill in: {', '.join(missing)}.")
        return
    ticket = {
        "ticket_id": ticket_id,
        "title": title,
        "description": description,
        "steps_to_reproduce": steps,
        "expected_result": expected,
        "actual_result": actual,
        "environment": environment,
        "priority": priority,
    }
    metadata = {"priority": priority} if priority else {}
    run = call(
        get_client().start_ticket_run, ticket, project=active_project(), metadata=metadata, pause_before_execution=gate
    )
    if run is not None:
        set_active_run(run["run_id"])
        st.switch_page(st.session_state["_pages"]["generate"])


def generate_test_cases() -> None:
    page_title("Generate Test Cases", "The agent analyzes the requirement, writes test cases and plans their execution.")

    def body(run: dict) -> None:
        nodes = {n["name"]: n for n in run["nodes"]}
        done = sum(nodes[n]["status"] == "completed" for n in DESIGN_NODES)
        running = [NODE_LABELS[n] for n in DESIGN_NODES if nodes[n]["status"] == "running"]
        st.progress(done / len(DESIGN_NODES), text=f"Design steps: {done}/{len(DESIGN_NODES)}" + (f" — {running[0]}…" if running else ""))
        pipeline(run)

        c1, c2, c3 = st.columns(3)
        c1.metric("Test cases", run["test_case_count"])
        c2.metric("Status", {"awaiting_execution": "Ready to execute"}.get(run["status"], run["status"].replace("_", " ").title()))
        c3.metric("Node errors", run["node_error_count"])

        design_errors = [e for e in run["errors"] if e["node"] in DESIGN_NODES]
        for error in design_errors:
            st.error(f"**{NODE_LABELS[error['node']]}** failed: {error['message']}", icon="🛑")

        analysis = call(get_client().requirement_analysis, run["run_id"])
        if analysis:
            st.subheader("Requirement analysis")
            st.markdown(f"**{analysis['feature']}** — {analysis['summary']}")
            left, right = st.columns(2)
            with left:
                _bullets("Acceptance criteria", analysis["acceptance_criteria"])
                _bullets("Business rules", analysis["business_rules"])
                _bullets("Actors", analysis["actors"])
            with right:
                _bullets("Edge cases", analysis["edge_cases"])
                _bullets("Risk areas", analysis["risk_areas"])
                _bullets("Ambiguities (not guessed)", analysis["ambiguities"])
        elif run["status"] == "running":
            st.caption("The requirement analysis appears here as soon as it is ready.")

        if run["test_case_count"]:
            next_step("Next: review the Test Cases", "test_cases")

    run_page(body)


def _bullets(title: str, items: list[str]) -> None:
    st.markdown(f"**{title}**")
    if items:
        st.markdown("\n".join(f"- {item}" for item in items))
    else:
        st.caption("None")
