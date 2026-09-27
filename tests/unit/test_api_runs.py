"""HTTP API over the workflow: start a run, pause before execution, review, report.

The graph uses the fake LLM and mock executor, so runs finish in milliseconds;
the tests poll ``GET /runs/{id}`` exactly as the Streamlit UI does.
"""

import time

import pytest
from fastapi.testclient import TestClient

from qa_agent.api.main import create_app
from qa_agent.graph import build_qa_graph
from qa_agent.graph.nodes.confidence_checker import make_confidence_checker
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from qa_agent.services.run_manager import RunManager
from qa_agent.services.run_service import RunService
from tests.conftest import PASSWORD_RESET_REQUIREMENT, password_reset_responses, repo_knowledge_base
from tests.fakes import FakeStructuredChatModel


def _client(*, failing=("TC-007",), nodes=None) -> TestClient:
    def factory() -> RunManager:
        graph = build_qa_graph(
            {
                "test_executor": make_mock_test_executor({tc: "failed" for tc in failing}),
                "confidence_checker": make_confidence_checker(threshold=0.7),
                **(nodes or {}),
            },
            llm=FakeStructuredChatModel(responses=password_reset_responses()),
            knowledge_base=repo_knowledge_base(),
        )
        return RunManager(RunService(graph))

    return TestClient(create_app(factory))


def _wait(client: TestClient, run_id: str, timeout_s: float = 20.0) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        run = client.get(f"/runs/{run_id}").json()
        if run["status"] != "running":
            return run
        time.sleep(0.05)
    raise AssertionError(f"run {run_id} still running after {timeout_s}s")


def _start(client: TestClient, **body) -> str:
    response = client.post("/runs", json={"requirement": PASSWORD_RESET_REQUIREMENT, "project": "shop", **body})
    assert response.status_code == 202, response.text
    return response.json()["run_id"]


def _nodes(run: dict) -> dict[str, str]:
    return {node["name"]: node["status"] for node in run["nodes"]}


def test_run_pauses_before_execution_until_execute_is_called():
    client = _client()
    run_id = _start(client)

    run = _wait(client, run_id)

    assert run["status"] == "awaiting_execution"
    assert run["test_case_count"] == 8
    assert run["test_counts"]["total"] == 0
    nodes = _nodes(run)
    assert [nodes[n] for n in ("requirement_analyzer", "test_case_generator", "test_planner")] == ["completed"] * 3
    assert nodes["test_executor"] == "waiting"
    assert nodes["final_report_generator"] == "pending"
    assert len(client.get(f"/runs/{run_id}/test-cases").json()) == 8
    assert client.get(f"/runs/{run_id}/test-plan").json()["entries"]
    assert client.get(f"/runs/{run_id}/requirement-analysis").json()["feature"]
    assert client.get(f"/runs/{run_id}/results").json() == []

    assert client.post(f"/runs/{run_id}/execute").status_code == 202
    run = _wait(client, run_id)

    assert run["status"] == "awaiting_review"
    assert run["test_counts"] == {"total": 8, "passed": 6, "failed": 1, "error": 0, "skipped": 1}
    assert _nodes(run)["human_review"] == "waiting"
    assert [f["test_case_id"] for f in client.get(f"/runs/{run_id}/failures").json()] == ["TC-007"]
    assert client.get(f"/runs/{run_id}/root-cause").json()["analysis"]["findings"]
    assert client.get(f"/runs/{run_id}/bugs").json()[0]["id"] == "BR-1"
    confidence = client.get(f"/runs/{run_id}/confidence").json()
    assert confidence["requires_human_review"] is True


def test_review_approval_completes_the_run_with_a_report():
    client = _client()
    run_id = _start(client, pause_before_execution=False)
    assert _wait(client, run_id)["status"] == "awaiting_review"
    review = client.get(f"/runs/{run_id}/review").json()
    assert review["pending"]["review_id"] == f"{run_id}-review-1"

    response = client.post(f"/runs/{run_id}/review", json={"decision": "approve", "reviewer": "arif"})
    assert response.status_code == 202
    run = _wait(client, run_id)

    assert (run["status"], run["outcome"]) == ("completed", "failed")
    assert client.get(f"/runs/{run_id}/review").json()["pending"] is None
    report = client.get(f"/runs/{run_id}/report").json()
    assert report["human_decision"] == "APPROVE"
    download = client.get(f"/runs/{run_id}/report.md")
    assert download.status_code == 200 and download.text == report["markdown"]


def test_request_reanalysis_loops_back_to_root_cause_analysis():
    client = _client()
    run_id = _start(client, pause_before_execution=False)
    _wait(client, run_id)

    client.post(f"/runs/{run_id}/review", json={"decision": "REQUEST_REANALYSIS", "comment": "Check token reuse"})
    run = _wait(client, run_id)

    assert run["status"] == "awaiting_review"
    visits = {node["name"]: node["visits"] for node in run["nodes"]}
    assert visits["root_cause_analyzer"] == 2
    assert len(client.get(f"/runs/{run_id}/review").json()["history"]) == 1


def test_all_passing_run_skips_the_failure_path():
    client = _client(failing=())
    run_id = _start(client, pause_before_execution=False)

    run = _wait(client, run_id)

    assert (run["status"], run["outcome"]) == ("completed", "passed")
    nodes = _nodes(run)
    assert nodes["final_report_generator"] == "completed"
    assert nodes["root_cause_analyzer"] == nodes["human_review"] == "skipped"
    assert client.get(f"/runs/{run_id}/report").json()["status"] == "passed"


def test_node_errors_are_reported_per_node():
    def broken(state):
        raise RuntimeError("planner exploded")

    client = _client(failing=(), nodes={"test_planner": broken})
    run_id = _start(client, pause_before_execution=False)

    run = _wait(client, run_id)

    assert run["status"] == "completed"
    assert _nodes(run)["test_planner"] == "error"
    assert run["errors"][0]["node"] == "test_planner"
    assert run["node_error_count"] == 1


@pytest.mark.parametrize(
    ("path", "body"),
    [("execute", None), ("review", {"decision": "APPROVE"})],
)
def test_actions_on_a_run_that_is_not_paused_there_conflict(path, body):
    client = _client(failing=())
    run_id = _start(client, pause_before_execution=False)
    _wait(client, run_id)

    response = client.post(f"/runs/{run_id}/{path}", json=body)

    assert response.status_code == 409
    assert "completed" in response.json()["detail"]


def test_invalid_review_decisions_are_rejected_without_resuming():
    client = _client()
    run_id = _start(client, pause_before_execution=False)
    _wait(client, run_id)

    missing_comment = client.post(f"/runs/{run_id}/review", json={"decision": "REQUEST_REANALYSIS"})
    unknown = client.post(f"/runs/{run_id}/review", json={"decision": "MAYBE"})

    assert missing_comment.status_code == unknown.status_code == 422
    assert client.get(f"/runs/{run_id}").json()["status"] == "awaiting_review"


def test_unknown_run_is_404():
    client = _client()
    for path in ("", "/test-cases", "/report", "/review"):
        assert client.get(f"/runs/missing{path}").status_code == 404
    assert client.post("/runs/missing/execute").status_code == 404


def test_blank_requirement_is_rejected():
    assert _client().post("/runs", json={"requirement": "   "}).status_code == 422


def test_projects_are_listed_and_created():
    client = _client()
    assert client.post("/projects", json={"name": "  Checkout  ", "description": "Payments"}).status_code == 201
    _wait(client, _start(client, project="shop"))

    projects = {p["name"]: p for p in client.get("/projects").json()}

    assert projects["Checkout"] == {"name": "Checkout", "description": "Payments", "run_count": 0}
    assert projects["shop"]["run_count"] == 1
    assert "default" in projects
    assert [r["project"] for r in client.get("/runs", params={"project": "shop"}).json()] == ["shop"]
    assert client.post("/projects", json={"name": " "}).status_code == 422


def test_crashed_run_reports_error_status():
    class ExplodingService(RunService):
        def start(self, *args, **kwargs):
            raise ConnectionError("database is down")

    graph = build_qa_graph(llm=FakeStructuredChatModel(responses=password_reset_responses()))
    client = TestClient(create_app(lambda: RunManager(ExplodingService(graph))))
    run_id = _start(client)

    run = _wait(client, run_id)

    assert run["status"] == "error"
    assert run["error"] == "ConnectionError: database is down"


def test_system_status_reports_configuration_without_secrets():
    body = _client().get("/system/status").json()
    assert {"llm_configured", "database_configured", "confidence_threshold", "warnings"} <= set(body)
    assert "openai_api_key" not in body


# --- chat ----------------------------------------------------------------------


class _RecordingChatModel:
    """Stands in for the chat model: records the prompt and returns a canned answer."""

    def __init__(self, answer="TC-007 failed because the mock executor failed it."):
        self.answer = answer
        self.calls = []

    def invoke(self, messages):
        from langchain_core.messages import AIMessage

        self.calls.append(messages)
        return AIMessage(self.answer)


def _chat_client(llm_factory):
    from qa_agent.api.dependencies import get_chat_model_factory

    client = _client()
    client.app.dependency_overrides[get_chat_model_factory] = lambda: llm_factory
    return client


def test_chat_answers_from_the_run_state():
    llm = _RecordingChatModel()
    client = _chat_client(lambda: llm)
    run_id = _start(client, pause_before_execution=False)
    _wait(client, run_id)

    response = client.post(
        f"/runs/{run_id}/chat",
        json={
            "message": "Why did TC-007 fail?",
            "history": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"answer": llm.answer}
    messages = llm.calls[0]
    assert "TC-007" in messages[0].content  # run data is in the system prompt
    assert [m.content for m in messages[1:]] == ["hi", "hello", "Why did TC-007 fail?"]


def test_chat_reports_missing_llm_configuration_as_503():
    from qa_agent.errors import LLMConfigurationError

    def no_key():
        raise LLMConfigurationError("OPENAI_API_KEY is not set")

    client = _chat_client(no_key)
    run_id = _start(client)
    _wait(client, run_id)

    response = client.post(f"/runs/{run_id}/chat", json={"message": "hi"})

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


def test_chat_validates_input_and_unknown_runs():
    client = _chat_client(lambda: _RecordingChatModel())

    assert client.post("/runs/nope/chat", json={"message": "hi"}).status_code == 404
    run_id = _start(client)
    _wait(client, run_id)
    assert client.post(f"/runs/{run_id}/chat", json={"message": "   "}).status_code == 422


def test_chat_web_app_is_served():
    client = _client()

    page = client.get("/app/")

    assert page.status_code == 200
    assert "QA Agent" in page.text
    assert client.get("/", follow_redirects=False).headers["location"] == "/app/"


# --- bug ticket verification -----------------------------------------------------

BUG_TICKET = {
    "ticket_id": "SD-77",
    "title": "Reset link can be reused",
    "steps_to_reproduce": ["Request a reset link", "Use it twice"],
    "expected_result": "The second use is rejected",
    "actual_result": "The password can be changed again",
}


def test_bug_ticket_run_reports_a_confirmed_verdict_when_the_reproduction_test_fails():
    client = _client(failing=("TC-001",))

    response = client.post("/runs", json={"bug_ticket": BUG_TICKET, "pause_before_execution": False})
    assert response.status_code == 202, response.text
    run = _wait(client, response.json()["run_id"])

    assert run["requirement"].startswith("Service desk bug ticket SD-77: Reset link can be reused")
    assert run["metadata"]["mode"] == "bug_verification"
    assert run["metadata"]["bug_ticket"]["steps_to_reproduce"] == ["Request a reset link", "Use it twice"]
    assert run["verdict"]["status"] == "confirmed"
    assert run["verdict"]["reproduction_test"] == "TC-001"
    assert run["verdict"]["failed_tests"] == ["TC-001"]


def test_bug_ticket_failing_variation_alone_does_not_confirm_the_bug():
    client = _client(failing=("TC-007",))

    run_id = client.post("/runs", json={"bug_ticket": BUG_TICKET, "pause_before_execution": False}).json()["run_id"]
    run = _wait(client, run_id)

    assert run["verdict"]["status"] == "not_reproduced"
    assert "TC-007" in run["verdict"]["explanation"]


def test_bug_ticket_run_is_not_reproduced_when_all_tests_pass():
    client = _client(failing=())

    run_id = client.post("/runs", json={"bug_ticket": BUG_TICKET, "pause_before_execution": False}).json()["run_id"]
    run = _wait(client, run_id)

    assert run["verdict"]["status"] == "not_reproduced"


def test_bug_ticket_verdict_is_pending_before_execution_and_absent_for_requirements():
    client = _client()

    ticket_run = _wait(client, client.post("/runs", json={"bug_ticket": BUG_TICKET}).json()["run_id"])
    requirement_run = _wait(client, _start(client))

    assert ticket_run["verdict"]["status"] == "pending"
    assert requirement_run["verdict"] is None


def test_bug_ticket_requires_expected_and_actual_results():
    client = _client()

    response = client.post("/runs", json={"bug_ticket": {"title": "Broken"}})

    assert response.status_code == 422
