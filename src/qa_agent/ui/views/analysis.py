"""Screens 10-13: root cause analysis, bug reports, confidence score, human review."""

import json

import pandas as pd
import streamlit as st

from qa_agent.ui.components import (
    SEVERITY,
    TONES,
    badge,
    call,
    empty_state,
    esc,
    fmt_time,
    get_client,
    next_step,
    page_title,
    pct,
    run_page,
    show_badges,
)

LIKELIHOOD = {
    "likely": ("Likely", "●", "serious"),
    "possible": ("Possible", "◐", "warning"),
    "speculative": ("Speculative", "○", "neutral"),
}
ORIGIN_LABELS = {
    "product_defect": "Product defect",
    "test_defect": "Test defect",
    "requirement_mismatch": "Requirement mismatch",
    "environment": "Environment",
    "unknown": "Unknown",
}
DECISIONS = {
    "APPROVE": "✓ Approve — publish the bug reports as proposed",
    "REJECT": "✕ Reject — discard the proposed actions",
    "REQUEST_REANALYSIS": "↻ Request reanalysis — rerun root cause analysis with my guidance",
}


def _no_failures_note(run: dict, what: str) -> None:
    if run["status"] == "completed" and not run["test_counts"]["failed"] and not run["test_counts"]["error"]:
        st.success(f"No failures in this run, so there is no {what}.", icon="✅")
    else:
        empty_state(f"The {what} appears here after failed tests are analyzed.")


def _evidence_table(evidence: list[dict]) -> None:
    observed = [e for e in evidence if e["kind"] == "observed"]
    knowledge = [e for e in evidence if e["kind"] == "knowledge_base"]
    left, right = st.columns(2)
    with left:
        st.markdown("**Observed evidence** (recorded by tests)")
        if observed:
            st.dataframe(
                pd.DataFrame([{"Fact": e["reference"], "Detail": e["detail"], "Interpretation": e.get("interpretation") or ""} for e in observed]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("None")
    with right:
        st.markdown("**Knowledge base references** (RAG)")
        if knowledge:
            st.dataframe(
                pd.DataFrame([{"Source": e["reference"], "Title": e["detail"], "Relevance": e.get("interpretation") or ""} for e in knowledge]),
                hide_index=True,
                width="stretch",
            )
        else:
            st.caption("None cited")


def root_cause() -> None:
    page_title(
        "Root Cause Analysis",
        "Probable causes are hypotheses with a likelihood, kept apart from observed evidence and open unknowns.",
    )

    def body(run: dict) -> None:
        data = call(get_client().root_cause, run["run_id"])
        if data is None:
            return
        analysis = data["analysis"]
        if not analysis:
            _no_failures_note(run, "root cause analysis")
            return

        st.markdown(f"**Summary.** {analysis['summary']}")
        st.metric("Overall confidence (lowest finding)", pct(analysis["confidence"]))

        for i, finding in enumerate(analysis["findings"], 1):
            cause = finding["probable_root_cause"]
            with st.container(border=True):
                show_badges(
                    badge(finding["severity_suggestion"], SEVERITY, label=f"Severity: {SEVERITY[finding['severity_suggestion']][0]}"),
                    badge(cause["likelihood"], LIKELIHOOD),
                    badge(None, label=ORIGIN_LABELS.get(finding["suspected_origin"], finding["suspected_origin"])),
                    badge(None, label=f"Confidence {pct(finding['confidence'])}"),
                )
                st.markdown(f"#### Finding {i}: {finding['summary']}")
                st.caption(f"Tests: {', '.join(finding['test_case_ids'])} · Component: {finding['affected_component']}")
                left, right = st.columns(2)
                left.markdown(f"**Expected**\n\n{finding['expected_behavior']}")
                right.markdown(f"**Observed**\n\n{finding['observed_behavior']}")
                st.markdown(f"**Probable root cause** · _{cause['likelihood']}_\n\n{cause['description']}")
                st.caption(f"Reasoning: {cause['reasoning']}")
                if finding["alternative_causes"]:
                    st.markdown("**Alternative causes**")
                    st.markdown("\n".join(f"- _{c['likelihood']}_: {c['description']}" for c in finding["alternative_causes"]))
                _evidence_table(finding["evidence"])
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Unknowns**")
                    st.markdown("\n".join(f"- {u}" for u in finding["unknowns"]) or "_None listed_")
                with c2:
                    st.markdown("**Recommended next investigation**")
                    st.markdown("\n".join(f"- {r}" for r in finding["recommended_next_investigation"]) or "_None_")
                if finding["adjustments"]:
                    st.warning("Guardrail corrections applied to the LLM output:\n\n" + "\n".join(f"- {a}" for a in finding["adjustments"]), icon="🛡️")

        if data["knowledge_usage"]:
            st.subheader("Knowledge base usage (RAG audit)")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Step": u["node"],
                            "Query": u["query"][:120],
                            "Retrieved": len(u["retrieved_sources"]),
                            "Cited": ", ".join(u["cited_sources"]) or "—",
                            "Discarded citations": ", ".join(u["discarded_citations"]) or "—",
                            "Note": u.get("message") or "",
                        }
                        for u in data["knowledge_usage"]
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        next_step("Next: Bug Report", "bugs")

    run_page(body)


def _bug_markdown(bug: dict) -> str:
    lines = [
        f"# {bug['id']}: {bug['title']}",
        "",
        f"- Severity: {bug['severity']} — {bug['severity_rationale']}",
        f"- Priority: {bug['priority']} — {bug['priority_rationale']}",
        f"- Type: {bug['report_type']}",
        f"- Component: {bug['affected_component']}",
        f"- Confidence: {pct(bug['confidence'])}",
        "",
        "## Summary",
        bug["summary"],
        "",
        "## Steps to reproduce",
        *[f"{i}. {s}" for i, s in enumerate(bug["reproduction_steps"], 1)],
        "",
        "## Expected result",
        bug["expected_result"],
        "",
        "## Actual result",
        bug["actual_result"],
        "",
        "## Probable root cause",
        f"{bug['probable_root_cause']['description']} ({bug['probable_root_cause']['likelihood']})",
    ]
    if bug["uncertainties"]:
        lines += ["", "## Uncertainties", *[f"- {u}" for u in bug["uncertainties"]]]
    return "\n".join(lines) + "\n"


def bug_reports() -> None:
    page_title(
        "Bug Report",
        "Facts (steps, expected/actual, evidence) come from recorded test data; the LLM writes only the narrative.",
    )

    def body(run: dict) -> None:
        bugs = call(get_client().bug_reports, run["run_id"])
        if bugs is None:
            return
        if not bugs:
            _no_failures_note(run, "bug report")
            return

        for bug in bugs:
            with st.container(border=True):
                show_badges(
                    badge(bug["severity"], SEVERITY, label=f"Severity: {SEVERITY[bug['severity']][0]}"),
                    badge(None, label=f"Priority {bug['priority'].upper()}"),
                    badge(None, label=bug["report_type"].replace("_", " ")),
                    badge(None, label=f"Revision {bug['revision']}"),
                    badge(None, label=f"Written by {bug['generated_by']}"),
                )
                st.markdown(f"### {bug['id']} · {bug['title']}")
                st.markdown(bug["summary"])
                st.caption(
                    f"Component: {bug['affected_component']} · Tests: {', '.join(bug['related_test_case_ids'])} · "
                    f"Confidence: {pct(bug['confidence'])}"
                )
                tab_repro, tab_evidence, tab_cause, tab_env = st.tabs(["Reproduce", "Evidence", "Root cause", "Environment"])
                with tab_repro:
                    if bug["preconditions"]:
                        st.markdown("**Preconditions**\n" + "\n".join(f"- {p}" for p in bug["preconditions"]))
                    st.markdown(f"**Steps** _(from {bug['reproduction_source'].replace('_', ' ')})_")
                    st.markdown("\n".join(f"{i}. {s}" for i, s in enumerate(bug["reproduction_steps"], 1)))
                    left, right = st.columns(2)
                    with left:
                        st.markdown("**Expected**")
                        st.info(bug["expected_result"])
                    with right:
                        st.markdown("**Actual**")
                        st.error(bug["actual_result"])
                with tab_evidence:
                    _evidence_table(bug["evidence"])
                with tab_cause:
                    cause = bug["probable_root_cause"]
                    st.markdown(f"**{cause['likelihood'].title()}:** {cause['description']}")
                    st.caption(cause["reasoning"])
                    st.markdown(f"**Severity rationale.** {bug['severity_rationale']}")
                    st.markdown(f"**Priority rationale.** {bug['priority_rationale']}")
                    if bug["uncertainties"]:
                        st.markdown("**Uncertainties**\n" + "\n".join(f"- {u}" for u in bug["uncertainties"]))
                    if bug["related_bugs"]:
                        st.markdown("**Similar past bugs (possible duplicates)**\n" + "\n".join(f"- `{b}`" for b in bug["related_bugs"]))
                    if bug["reviewer_feedback"]:
                        st.markdown("**Reviewer feedback**\n" + "\n".join(f"- {f}" for f in bug["reviewer_feedback"]))
                with tab_env:
                    env = bug["environment"]
                    st.markdown(
                        f'<dl class="qa-kv"><dt>App env</dt><dd>{esc(env["app_env"])}</dd>'
                        f'<dt>Targets</dt><dd>{esc(", ".join(env["targets"]) or "—")}</dd>'
                        f'<dt>Automation</dt><dd>{esc(", ".join(env["automation_types"]) or "—")}</dd>'
                        f'<dt>Recorded</dt><dd>{esc(fmt_time(env["recorded_at"]))}</dd></dl>',
                        unsafe_allow_html=True,
                    )
                    for unknown in env["unknowns"]:
                        st.caption(f"Unknown: {unknown}")
                d1, d2 = st.columns(2)
                d1.download_button("Download .md", _bug_markdown(bug), file_name=f"{bug['id']}.md", mime="text/markdown", key=f"md-{bug['id']}")
                d2.download_button("Download .json", json.dumps(bug, indent=2), file_name=f"{bug['id']}.json", mime="application/json", key=f"js-{bug['id']}")
        next_step("Next: Confidence Score", "confidence")

    run_page(body)


def _meter(score: float, threshold: float) -> None:
    tone = TONES["good"] if score >= threshold else TONES["warning"]
    st.markdown(
        f'<div class="qa-meter" role="img" aria-label="Confidence {score:.0%}, threshold {threshold:.0%}">'
        f'<div class="fill" style="width:{score * 100:.1f}%;--tone:{tone}"></div>'
        f'<div class="mark" style="left:{threshold * 100:.1f}%"><span>threshold {threshold:.0%}</span></div></div>',
        unsafe_allow_html=True,
    )


def confidence() -> None:
    page_title("Confidence Score", "Decides whether the agent may finish on its own or must ask a human.")

    def body(run: dict) -> None:
        client = get_client()
        score = call(client.confidence, run["run_id"])
        if score is None:
            _no_failures_note(run, "confidence check")
            return

        c1, c2, c3 = st.columns(3)
        c1.metric("Confidence", pct(score["score"]))
        c2.metric("Threshold", pct(score["threshold"]))
        c3.metric("Human review", "Required" if score["requires_human_review"] else "Not required")
        _meter(score["score"], score["threshold"])
        if score["requires_human_review"]:
            st.warning("The run needs a human decision before it can finish.", icon="⏸")
        else:
            st.success("Confidence is above the threshold and no sensitive action is proposed.", icon="✅")
        st.markdown("**Why**")
        st.markdown("\n".join(f"- {r}" for r in score["reasons"]) or "_No reasons recorded_")

        analysis = (call(client.root_cause, run["run_id"]) or {}).get("analysis")
        if analysis and analysis["findings"]:
            st.markdown("**Per-finding confidence**")
            for finding in analysis["findings"]:
                st.caption(f"{', '.join(finding['test_case_ids'])} — {finding['summary']}")
                _meter(finding["confidence"], score["threshold"])
        next_step("Next: Human Review" if score["requires_human_review"] else "Next: Final QA Report",
                  "review" if score["requires_human_review"] else "report")

    run_page(body)


def _review_card(review: dict) -> None:
    decided = review.get("human_decision")
    st.markdown(
        f"**{esc(review['review_id'])}** · requested {fmt_time(review['requested_at'])}"
        + (f" · decision **{decided}** by {review.get('reviewer') or 'anonymous'} at {fmt_time(review.get('decided_at'))}" if decided else "")
    )
    if review.get("reviewer_comment"):
        st.caption(f"Comment: {review['reviewer_comment']}")


def human_review() -> None:
    page_title("Human Review", "Low confidence or sensitive actions pause the run until a person decides.")

    def body(run: dict) -> None:
        data = call(get_client().review, run["run_id"])
        if data is None:
            return
        pending = data["pending"]

        if pending:
            st.markdown("#### Waiting for your decision")
            with st.container(border=True):
                show_badges(badge("awaiting_review"), badge(None, label=f"Confidence {pct(pending['confidence'])}"))
                st.markdown("**Why the run paused**")
                st.markdown("\n".join(f"- {r}" for r in pending["reasons"]) or pending["reason"])
                st.markdown(f"**AI recommendation.** {pending['ai_recommendation']}")
                st.markdown(f"**Proposed action.** {pending['proposed_action']}")
                if pending["sensitive_actions"]:
                    st.warning("Needs approval by policy: " + ", ".join(a.replace("_", " ") for a in pending["sensitive_actions"]), icon="🔐")
                with st.expander("Evidence"):
                    _evidence_table(pending["evidence"])

            with st.form("review-decision"):
                decision = st.radio("Decision", list(DECISIONS), format_func=DECISIONS.get)
                comment = st.text_area(
                    "Comment",
                    placeholder="Required for reanalysis: what should the analysis reconsider?",
                )
                reviewer = st.text_input("Reviewer", value=st.session_state.get("reviewer", ""))
                if st.form_submit_button("Submit decision", type="primary", width="stretch"):
                    if decision == "REQUEST_REANALYSIS" and not comment.strip():
                        st.warning("Add a comment telling the analysis what to reconsider.")
                    else:
                        st.session_state["reviewer"] = reviewer
                        if call(get_client().submit_review, run["run_id"], decision, comment.strip(), reviewer.strip()) is not None:
                            st.toast(f"Decision submitted: {decision}")
                            st.rerun()
        elif run["status"] == "running" and data["latest"] and data["latest"].get("human_decision"):
            st.info("Decision received — the workflow is continuing.", icon="⏳")
        elif data["latest"] and data["latest"].get("human_decision"):
            st.success(f"Reviewed: **{data['latest']['human_decision']}**.", icon="✅")
        else:
            empty_state("No review is pending. Runs pause here only when confidence is low or an action is sensitive.")

        reviews = [*data["history"]]
        if data["latest"] and data["latest"].get("human_decision") and all(r["review_id"] != data["latest"]["review_id"] for r in reviews):
            reviews.append(data["latest"])
        if reviews:
            st.subheader("Review history")
            for review in reviews:
                with st.container(border=True):
                    _review_card(review)
        if not pending and run["status"] == "completed":
            next_step("Next: Final QA Report", "report")

    run_page(body)
