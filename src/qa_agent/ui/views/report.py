"""Screen 14: the final QA report."""

import json

import pandas as pd
import streamlit as st

from qa_agent.ui.components import (
    SEVERITY,
    badge,
    call,
    empty_state,
    get_client,
    page_title,
    pct,
    run_page,
    show_badges,
)


def final_report() -> None:
    page_title("Final QA Report", "Summary of the run: outcome, metrics, bugs, confidence and the human decision.")

    def body(run: dict) -> None:
        report = call(get_client().report, run["run_id"])
        if report is None:
            if run["status"] in ("awaiting_execution", "awaiting_review"):
                empty_state("The report is generated when the run finishes. It is waiting for approval right now.", icon="⏸")
            elif run["status"] == "error":
                st.error("The run stopped before a report could be generated.", icon="🛑")
            else:
                empty_state("The report appears here when the run finishes.")
            return

        extra = [
            *([badge(None, label=f"Human decision: {report['human_decision']}")] if report.get("human_decision") else []),
            *([badge("error", label=f"{report['error_count']} node errors")] if report["error_count"] else []),
        ]
        if extra:
            show_badges(*extra)
        st.markdown(f"### {report['summary']}")

        m = st.columns(6)
        m[0].metric("Executed", report["total_tests"])
        m[1].metric("✓ Passed", report["passed"])
        m[2].metric("✕ Failed", report["failed"])
        m[3].metric("– Skipped", report["skipped"])
        m[4].metric("Pass rate", pct(report["pass_rate"]))
        confidence = report.get("confidence_score")
        m[5].metric("Confidence", pct(confidence["score"]) if confidence else "—")

        if report["bug_reports"]:
            st.subheader("Bugs")
            severities = [b["severity"] for b in report["bug_reports"]]
            show_badges(*(badge(s, SEVERITY, label=f"{SEVERITY[s][0]}: {severities.count(s)}") for s in SEVERITY if s in severities))
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "ID": b["id"],
                            "Title": b["title"],
                            "Severity": b["severity"],
                            "Priority": b["priority"].upper(),
                            "Type": b["report_type"].replace("_", " "),
                            "Component": b["affected_component"],
                            "Confidence": pct(b["confidence"]),
                        }
                        for b in report["bug_reports"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )

        st.subheader("Report")
        with st.container(border=True):
            st.markdown(report["markdown"] or "_Empty report_")

        d1, d2 = st.columns(2)
        d1.download_button(
            "Download report (.md)",
            report["markdown"],
            file_name=f"qa-report-{run['run_id'][:8]}.md",
            mime="text/markdown",
            type="primary",
            width="stretch",
        )
        d2.download_button(
            "Download report (.json)",
            json.dumps(report, indent=2),
            file_name=f"qa-report-{run['run_id'][:8]}.json",
            mime="application/json",
            width="stretch",
        )

    run_page(body)
