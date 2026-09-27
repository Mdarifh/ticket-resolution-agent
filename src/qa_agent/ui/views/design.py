"""Screens 4-5: generated test cases and the execution plan."""

import pandas as pd
import streamlit as st

from qa_agent.ui.components import call, empty_state, get_client, next_step, page_title, run_page

AUTOMATION_LABELS = {"api": "API", "ui": "UI (browser)", "manual": "Manual"}


def test_cases() -> None:
    page_title("Test Cases", "Generated from the requirement analysis and the QA knowledge base.")

    def body(run: dict) -> None:
        cases = call(get_client().test_cases, run["run_id"])
        if cases is None:
            return
        if not cases:
            empty_state("No test cases yet." if run["status"] == "running" else "The run produced no test cases.")
            return

        f1, f2, f3 = st.columns(3)
        types = f1.multiselect("Type", sorted({c["test_type"] for c in cases}), key="tc-type")
        priorities = f2.multiselect("Priority", ["high", "medium", "low"], key="tc-priority")
        automations = f3.multiselect("Automation", sorted({c["automation_type"] for c in cases}), key="tc-auto")
        shown = [
            c
            for c in cases
            if (not types or c["test_type"] in types)
            and (not priorities or c["priority"] in priorities)
            and (not automations or c["automation_type"] in automations)
        ]

        m = st.columns(4)
        m[0].metric("Test cases", len(cases))
        m[1].metric("API", sum(c["automation_type"] == "api" for c in cases))
        m[2].metric("UI", sum(c["automation_type"] == "ui" for c in cases))
        m[3].metric("Manual", sum(c["automation_type"] == "manual" for c in cases))

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "ID": c["test_case_id"],
                        "Title": c["title"],
                        "Type": c["test_type"].replace("_", " "),
                        "Priority": c["priority"],
                        "Automation": AUTOMATION_LABELS.get(c["automation_type"], c["automation_type"]),
                        "Steps": len(c["steps"]),
                    }
                    for c in shown
                ]
            ),
            hide_index=True,
            width="stretch",
        )

        st.subheader("Details")
        for case in shown:
            with st.expander(f"{case['test_case_id']} · {case['title']}"):
                st.markdown(case["description"])
                left, right = st.columns(2)
                with left:
                    st.markdown("**Preconditions**")
                    st.markdown("\n".join(f"- {p}" for p in case["preconditions"]) or "_None_")
                    st.markdown("**Test data**")
                    if case["test_data"]:
                        st.dataframe(pd.DataFrame(case["test_data"]), hide_index=True, width="stretch")
                    else:
                        st.caption("None")
                with right:
                    st.markdown("**Steps**")
                    st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(case["steps"], 1)))
                    st.markdown("**Expected result**")
                    st.success(case["expected_result"])
        next_step("Next: Test Plan", "test_plan")

    run_page(body)


def test_plan() -> None:
    page_title("Test Plan", "How and in which order each test case runs.")

    def body(run: dict) -> None:
        client = get_client()
        plan = call(client.test_plan, run["run_id"])
        if plan is None:
            if run["status"] == "running":
                empty_state("The plan is being prepared.")
            else:
                empty_state("No test plan was produced. Check **Test Run Status** for node errors.")
            return
        cases = {c["test_case_id"]: c for c in call(client.test_cases, run["run_id"]) or []}

        st.markdown(f"**Strategy.** {plan['strategy']}")
        for warning in plan["warnings"]:
            st.warning(warning, icon="⚠️")

        entries = sorted(plan["entries"], key=lambda e: e["execution_order"])
        counts = {kind: sum(e["automation_type"] == kind for e in entries) for kind in AUTOMATION_LABELS}
        m = st.columns(4)
        m[0].metric("Planned tests", len(entries))
        m[1].metric("API tests", counts["api"])
        m[2].metric("UI tests", counts["ui"])
        m[3].metric("Manual (skipped by the agent)", counts["manual"])

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Order": e["execution_order"],
                        "ID": e["test_case_id"],
                        "Title": cases.get(e["test_case_id"], {}).get("title", ""),
                        "Runs as": AUTOMATION_LABELS.get(e["automation_type"], e["automation_type"]),
                        "Rationale": e["rationale"],
                    }
                    for e in entries
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        next_step("Next: Execute Tests", "execute")

    run_page(body)
