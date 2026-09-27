"""Screens 6-9: execute tests, run status, results, failure details."""

import pandas as pd
import streamlit as st

from qa_agent.ui.api_client import BackendError
from qa_agent.ui.components import (
    NODE_LABELS,
    NODE_STATUS,
    TEST_STATUS,
    TONES,
    badge,
    call,
    empty_state,
    esc,
    fmt_ms,
    fmt_time,
    get_client,
    next_step,
    page_title,
    pipeline,
    run_page,
    show_badges,
)

EXECUTION_NODES = ("test_executor", "result_analyzer")


def _status_text(status: str, mapping=TEST_STATUS) -> str:
    text, icon, _ = mapping.get(status, (status, "•", "neutral"))
    return f"{icon} {text}"


def execute_tests() -> None:
    page_title("Execute Tests", "Run the planned API and UI tests against the configured test environment.")

    def body(run: dict) -> None:
        client = get_client()
        nodes = {n["name"]: n for n in run["nodes"]}
        executed = nodes["test_executor"]["status"] in ("completed", "error")

        if run["status"] == "awaiting_execution":
            plan = call(client.test_plan, run["run_id"]) or {"entries": []}
            system = call(client.system_status) or {}
            kinds = [e["automation_type"] for e in plan["entries"]]
            st.markdown("#### Approval required")
            st.markdown(
                "The agent has finished designing the tests and is **waiting for approval** before it sends "
                "requests to the system under test."
            )
            m = st.columns(3)
            m[0].metric("API tests", kinds.count("api"))
            m[1].metric("UI tests", kinds.count("ui"))
            m[2].metric("Manual (will be skipped)", kinds.count("manual"))
            with st.container(border=True):
                st.markdown(
                    f'<dl class="qa-kv"><dt>API target</dt><dd>{esc(system.get("api_test_base_url") or "not configured")}</dd>'
                    f'<dt>UI target</dt><dd>{esc(system.get("ui_test_base_url") or "not configured")}</dd>'
                    f'<dt>Environment</dt><dd>{esc(system.get("app_env", "?"))}</dd></dl>',
                    unsafe_allow_html=True,
                )
            if kinds.count("api") and not system.get("api_test_base_url"):
                st.warning("API_TEST_BASE_URL is not set on the backend: API tests will be reported as errors.", icon="⚠️")
            if kinds.count("ui") and not system.get("ui_test_base_url"):
                st.warning("UI_TEST_BASE_URL is not set on the backend: UI tests will be reported as errors.", icon="⚠️")
            if st.button("▶ Execute Tests", type="primary", width="stretch"):
                if call(client.execute, run["run_id"]) is not None:
                    st.rerun()
            return

        if run["status"] == "running" and not executed:
            if any(nodes[n]["status"] == "running" for n in EXECUTION_NODES):
                st.info("Executing tests… results appear as soon as the run is analyzed.", icon="⏳")
            else:
                st.info("The agent is still designing the tests. Execution starts when planning is done"
                        + (" and you approve it here." if run["pause_before_execution"] else "."), icon="⏳")
            pipeline(run)
            return

        if executed:
            counts = run["test_counts"]
            st.success(
                f"Execution finished: {counts['passed']} passed, {counts['failed']} failed, "
                f"{counts['error']} errors, {counts['skipped']} skipped.",
                icon="✅",
            )
            pipeline(run)
            next_step("Next: Test Results", "results")
        else:
            empty_state("Tests were not executed in this run. See **Test Run Status** for details.")

    run_page(body)


def run_status() -> None:
    page_title("Test Run Status", "Live progress of the LangGraph workflow, node by node.")

    def body(run: dict) -> None:
        pipeline(run)
        left, right = st.columns([3, 2])
        with left:
            st.markdown("#### Timeline")
            rows = [
                {
                    "Step": NODE_LABELS.get(n["name"], n["name"]),
                    "Status": _status_text(n["status"], NODE_STATUS),
                    "Runs": n["visits"],
                    "Started": fmt_time(n["started_at"]),
                    "Finished": fmt_time(n["finished_at"]),
                    "Duration": fmt_ms(n["duration_ms"]),
                }
                for n in run["nodes"]
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        with right:
            st.markdown("#### Run")
            meta = "".join(f"<dt>{esc(k)}</dt><dd>{esc(v)}</dd>" for k, v in run["metadata"].items())
            st.markdown(
                f'<div class="qa-card"><dl class="qa-kv">'
                f'<dt>Run id</dt><dd><code>{esc(run["run_id"])}</code></dd>'
                f'<dt>Project</dt><dd>{esc(run["project"])}</dd>'
                f'<dt>Started</dt><dd>{esc(fmt_time(run["created_at"]))}</dd>'
                f'<dt>Last update</dt><dd>{esc(fmt_time(run["updated_at"]))}</dd>'
                f'<dt>Execution gate</dt><dd>{"on" if run["pause_before_execution"] else "off"}</dd>'
                f"{meta}</dl></div>",
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Tests executed", run["test_counts"]["total"])
            m2.metric("Bug reports", run["bug_count"])
        st.markdown("#### Node errors")
        if not run["errors"]:
            st.caption("No node reported an error.")
        for error in run["errors"]:
            st.error(
                f"**{NODE_LABELS.get(error['node'], error['node'])}** · {error['error_type']} "
                f"at {fmt_time(error['occurred_at'])}\n\n{error['message']}",
                icon="🛑",
            )
        if run["errors"]:
            st.caption("A failing step does not stop the workflow: it continues and the final report records the error.")
        _observability(run)

    run_page(body)


@st.cache_data(ttl=300, show_spinner=False)
def _trace_links(run_id: str, trace_count: int) -> dict[str, str]:
    """trace_id -> LangSmith URL (resolving links calls LangSmith, so cache it)."""
    traces = get_client().traces(run_id)
    return {t["trace_id"]: t["url"] for t in traces if t.get("url")}


def _observability(run: dict) -> None:
    with st.expander("🔭 Observability — trace this run across UI, API, LangGraph, tools and DB"):
        short = run["run_id"][:8]
        st.markdown(
            f'<dl class="qa-kv"><dt>QA run id</dt><dd><code>{esc(run["run_id"])}</code></dd>'
            f"<dt>Backend / UI logs</dt><dd>lines tagged <code>run={esc(short)}</code></dd>"
            f"<dt>LangGraph</dt><dd>thread_id <code>{esc(run['run_id'])}</code></dd>"
            f"<dt>Database</dt><dd><code>agent_runs.id = '{esc(run['run_id'])}'</code></dd>"
            f"<dt>LangSmith</dt><dd>"
            + (
                f"project <code>{esc(run['langsmith_project'])}</code>, filter metadata "
                f"<code>qa_run_id = {esc(run['run_id'])}</code>"
                if run.get("langsmith_tracing")
                else "tracing off (see README → Observability)"
            )
            + "</dd></dl>",
            unsafe_allow_html=True,
        )
        if not run.get("traces"):
            st.caption("No workflow segment has started yet.")
            return
        links: dict[str, str] = {}
        if run.get("langsmith_tracing"):
            try:
                links = _trace_links(run["run_id"], len(run["traces"]))
            except BackendError:
                links = {}
        st.markdown("**Traces** — one per workflow segment")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Segment": f"qa_run.{t['segment']}",
                        "Trace id": t["trace_id"],
                        "Started": fmt_time(t["started_at"]),
                        "LangSmith": links.get(t["trace_id"], ""),
                    }
                    for t in run["traces"]
                ]
            ),
            column_config={"LangSmith": st.column_config.LinkColumn("LangSmith", display_text="Open trace")},
            hide_index=True,
            width="stretch",
        )


def _proportion_bar(counts: dict[str, int]) -> None:
    total = sum(counts.values())
    if not total:
        return
    segments = "".join(
        f'<div title="{TEST_STATUS[s][0]}: {n}" style="flex:{n};background:{TONES[TEST_STATUS[s][2]]};height:100%;border-radius:4px"></div>'
        for s, n in counts.items()
        if n
    )
    legend = " ".join(badge(s, TEST_STATUS, label=f"{TEST_STATUS[s][0]} {n}") for s, n in counts.items() if n)
    st.markdown(
        f'<div style="display:flex;gap:2px;height:14px;margin:.3rem 0 .5rem 0">{segments}</div>{legend}',
        unsafe_allow_html=True,
    )


def test_results() -> None:
    page_title("Test Results", "Outcome of every executed test, with the evidence the tools recorded.")

    def body(run: dict) -> None:
        client = get_client()
        results = call(client.results, run["run_id"])
        if results is None:
            return
        if not results:
            empty_state("No results yet. Tests run after planning" + (" and approval on **Execute Tests**." if run["pause_before_execution"] else "."))
            return
        titles = {c["test_case_id"]: c["title"] for c in call(client.test_cases, run["run_id"]) or []}
        counts = {s: sum(r["status"] == s for r in results) for s in TEST_STATUS}
        executed = counts["passed"] + counts["failed"] + counts["error"]

        m = st.columns(6)
        m[0].metric("Total", len(results))
        m[1].metric("✓ Passed", counts["passed"])
        m[2].metric("✕ Failed", counts["failed"])
        m[3].metric("! Errors", counts["error"])
        m[4].metric("– Skipped", counts["skipped"])
        m[5].metric("Pass rate", f"{counts['passed'] / executed:.0%}" if executed else "—", help="Passed / executed (skipped excluded)")
        _proportion_bar(counts)

        status_filter = st.segmented_control(
            "Show", ["All", "Passed", "Failed", "Error", "Skipped"], default="All", key="res-filter"
        )
        shown = [r for r in results if status_filter in (None, "All") or r["status"] == status_filter.lower()]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": r["test_case_id"],
                        "Title": titles.get(r["test_case_id"], ""),
                        "Status": _status_text(r["status"]),
                        "Runs as": r["evidence"].get("automation_type", ""),
                        "Duration": fmt_ms(r["duration_ms"]),
                        "Message": r.get("message") or "",
                    }
                    for r in shown
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        for result in shown:
            with st.expander(f"{_status_text(result['status'])} · {result['test_case_id']} · {titles.get(result['test_case_id'], '')}"):
                if result.get("message"):
                    st.markdown(f"**Message:** {result['message']}")
                st.markdown("**Evidence** (recorded by the test tool)")
                st.json(result["evidence"], expanded=False)
        if counts["failed"] or counts["error"]:
            next_step("Next: Failure Details", "failures")
        else:
            next_step("Next: Final QA Report", "report")

    run_page(body)


def failure_details() -> None:
    page_title("Failure Details", "Facts extracted from each failed test. Observed data only — no interpretation yet.")

    def body(run: dict) -> None:
        failures = call(get_client().failures, run["run_id"])
        if failures is None:
            return
        if not failures:
            if run["test_counts"]["total"]:
                st.success("No failures: every executed test passed.", icon="✅")
            else:
                empty_state("Failures appear here once tests have run.")
            return
        st.metric("Failed or errored tests", len(failures))
        for failure in failures:
            with st.container(border=True):
                show_badges(
                    badge(failure["status"], TEST_STATUS),
                    badge(None, label=(failure.get("category") or "uncategorized").replace("_", " ")),
                    badge(None, label=(failure.get("automation_type") or "?").upper()),
                )
                st.markdown(f"#### {failure['test_case_id']} · {failure.get('test_title') or ''}")
                if failure.get("signal"):
                    st.markdown(f"**Key signal:** `{failure['signal']}`")
                left, right = st.columns(2)
                with left:
                    st.markdown("**Expected**")
                    st.info(failure.get("expected_result") or "—")
                with right:
                    st.markdown("**Actual**")
                    st.error(failure.get("actual_result") or failure.get("message") or "—")
                if failure["observed_facts"]:
                    st.markdown("**Observed facts**")
                    st.dataframe(
                        pd.DataFrame(
                            [{"Fact": f["id"], "Source": f["source"], "Detail": f["detail"]} for f in failure["observed_facts"]]
                        ),
                        hide_index=True,
                        width="stretch",
                    )
        next_step("Next: Root Cause Analysis", "root_cause")

    run_page(body)
