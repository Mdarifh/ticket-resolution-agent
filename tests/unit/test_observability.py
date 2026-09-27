"""Observability: secret redaction, correlation ids in logs, and LangSmith traces.

Traces are captured with a LangSmith client whose HTTP session is a mock, so
the test sees exactly the run payloads that would be sent to LangSmith.
"""

import io
import json
import logging
import time
from unittest.mock import MagicMock

import httpx
import pytest
from fastapi.testclient import TestClient
from langsmith import Client
from langsmith.run_helpers import tracing_context
from pydantic import SecretStr

from qa_agent.api.main import create_app
from qa_agent.config import Settings
from qa_agent.graph import build_qa_graph
from qa_agent.graph.nodes.test_executor import make_mock_test_executor
from qa_agent.logging_conf import ContextFilter, build_handler
from qa_agent.observability import context, redaction, tracing
from qa_agent.observability.redaction import redact, redact_text, safe_url
from qa_agent.services.run_manager import RunManager
from qa_agent.services.run_service import RunService
from qa_agent.tools.api_test_tool import ApiTestConfig, ApiTestExecutor
from tests.conftest import PASSWORD_RESET_REQUIREMENT, fake_ui_runner, password_reset_responses, repo_knowledge_base
from tests.fakes import FakeStructuredChatModel

OPENAI_KEY = "sk-proj-abcdefghijklmnopqrstuvwx"
LANGSMITH_KEY = "lsv2_pt_0123456789abcdef0123456789"
DB_PASSWORD = "s3cret-db-pass"


@pytest.fixture
def configured_secrets(monkeypatch):
    monkeypatch.setattr(redaction, "_configured_secrets", lambda: ("tok-configured-123", DB_PASSWORD))


# --- redaction -------------------------------------------------------------------


def test_redact_text_masks_configured_secrets_and_known_credential_shapes(configured_secrets):
    text = (
        f"key={OPENAI_KEY} ls={LANGSMITH_KEY} Authorization: Bearer abc.def.ghi-123 "
        f"url=postgresql+psycopg://qa:{DB_PASSWORD}@db:5432/qa token tok-configured-123 api_key=xyz12345"
    )

    redacted = redact_text(text)

    for secret in (OPENAI_KEY, LANGSMITH_KEY, "abc.def.ghi-123", DB_PASSWORD, "tok-configured-123", "xyz12345"):
        assert secret not in redacted
    assert "postgresql+psycopg://qa:***@db:5432/qa" in redacted


def test_redact_walks_payloads_and_keeps_test_data(configured_secrets):
    payload = {
        "headers": {"Authorization": "Bearer live-token-999", "Accept": "application/json"},
        "body": {"email": "registered.user@example.com", "token": "used-token-456", "new_password": "An0ther-Passw0rd!"},
        "api_key": SecretStr("anything"),
        "items": [{"cookie": "session=1"}, f"note {OPENAI_KEY}"],
    }

    cleaned = redact(payload)

    assert cleaned["headers"] == {"Authorization": "***", "Accept": "application/json"}
    assert cleaned["api_key"] == "***"
    assert cleaned["items"] == [{"cookie": "***"}, "note sk-***"]
    # Test data is what a failed test needs to be debugged: not a secret.
    assert cleaned["body"] == payload["body"]


def test_safe_url_drops_query_and_credentials():
    assert safe_url("https://user:pw123@sut.local/api?token=abc") == "https://user:***@sut.local/api"


# --- logging ---------------------------------------------------------------------


def _capture(log_format: str = "text") -> tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = build_handler(log_format)
    handler.setStream(stream)
    logger = logging.getLogger(f"test.observability.{log_format}")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    return logger, stream


def test_log_lines_carry_correlation_ids_and_never_secrets(configured_secrets):
    logger, stream = _capture()

    with context.bind(run_id="5455144eec7c4b1f", request_id="9f2c01abffff", node="test_executor"):
        logger.info("calling with %s", OPENAI_KEY)
        try:
            raise RuntimeError(f"connect failed postgresql://qa:{DB_PASSWORD}@db/qa")
        except RuntimeError:
            logger.exception("boom")
    logger.info("outside")

    out = stream.getvalue()
    assert "run=5455144e req=9f2c01ab node=test_executor" in out
    assert OPENAI_KEY not in out and DB_PASSWORD not in out
    assert "sk-***" in out and "qa:***@db" in out
    assert out.strip().splitlines()[-1].split(" | ")[2] == "-"  # no ids bound outside the block


def test_json_log_format_includes_context_fields(configured_secrets):
    logger, stream = _capture("json")

    with context.bind(run_id="abc12345", client_session="sess1234"):
        logger.warning("key %s", OPENAI_KEY)

    entry = json.loads(stream.getvalue())
    assert entry["run_id"] == "abc12345" and entry["client_session"] == "sess1234"
    assert entry["level"] == "WARNING" and OPENAI_KEY not in entry["message"]


def test_bind_rejects_unknown_fields():
    with pytest.raises(ValueError):
        with context.bind(user="x"):
            pass


# --- tracing configuration -------------------------------------------------------


@pytest.fixture
def restore_tracing(monkeypatch):
    for name in ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2", "LANGSMITH_API_KEY", "LANGSMITH_PROJECT"):
        monkeypatch.setenv(name, "placeholder")  # so monkeypatch restores whatever configure_tracing writes
    yield
    tracing.configure_tracing(Settings(_env_file=None))


def test_tracing_is_exported_to_the_environment_when_configured(restore_tracing):
    import os

    status = tracing.configure_tracing(
        Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key=LANGSMITH_KEY, langsmith_project="qa-demo")
    )

    assert status.enabled and status.project == "qa-demo"
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == LANGSMITH_KEY
    assert os.environ["LANGSMITH_PROJECT"] == "qa-demo"


def test_tracing_without_a_key_is_disabled_everywhere(restore_tracing, caplog):
    import os

    with caplog.at_level(logging.WARNING):
        status = tracing.configure_tracing(Settings(_env_file=None, langsmith_tracing=True, langsmith_api_key=None))

    assert not status.enabled and "LANGSMITH_API_KEY" in status.reason
    assert os.environ["LANGSMITH_TRACING"] == os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert "tracing requested but disabled" in caplog.text
    assert tracing.trace_url("00000000-0000-0000-0000-000000000000") is None


# --- LangSmith traces ------------------------------------------------------------


class CapturedTraces:
    """Runs posted to a mocked LangSmith API, merged with their end-of-run patches."""

    def __init__(self) -> None:
        self.session = MagicMock()
        self.session.request.return_value.status_code = 200
        self.client = Client(session=self.session, api_key="lsv2_test_key_000000", api_url="http://langsmith.test", auto_batch_tracing=False)

    def runs(self) -> list[dict]:
        self.client.flush() if hasattr(self.client, "flush") else None
        by_id: dict[str, dict] = {}
        for call in self.session.request.call_args_list:
            method, url = call.args[0], call.args[1]
            data = call.kwargs.get("data")
            if not data or "/runs" not in url:
                continue
            body = json.loads(data)
            run_id = body.get("id") or url.rstrip("/").split("/")[-1]
            by_id.setdefault(run_id, {}).update(body)
        return list(by_id.values())


def _password_reset_server(request: httpx.Request) -> httpx.Response:
    if request.url.path.endswith("/confirm"):
        return httpx.Response(200, json={"message": "Password changed"})  # re-creates the token reuse bug
    return httpx.Response(202, json={"message": "If the email is registered, a reset link has been sent."})


def test_one_run_segment_is_one_trace_covering_graph_llm_rag_tools_and_failures():
    traces = CapturedTraces()
    api = ApiTestExecutor(
        ApiTestConfig(base_url="https://sut.test.local"),
        httpx.Client(transport=httpx.MockTransport(_password_reset_server)),
    )

    def broken_confidence_checker(state):
        raise RuntimeError("confidence service unavailable")

    graph = build_qa_graph(
        {"confidence_checker": broken_confidence_checker},
        llm=FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
        api_executor=api,
        ui_runner=fake_ui_runner(),
    )
    recorded: list[tracing.TraceRecord] = []

    with tracing_context(enabled=True, client=traces.client, project_name="qa-test"):
        with context.bind(project="shop", request_id="req-123", on_trace=recorded.append):
            outcome = RunService(graph).start(PASSWORD_RESET_REQUIREMENT, run_id="run0001abcd")
    time.sleep(0.5)
    runs = traces.runs()

    [segment] = recorded
    assert segment.segment == "start"
    root = next(r for r in runs if r.get("id") == segment.trace_id)
    assert root["name"] == "qa_run.start"
    meta = root["extra"]["metadata"]
    assert (meta["qa_run_id"], meta["thread_id"], meta["project"], meta["request_id"]) == ("run0001abcd", "run0001abcd", "shop", "req-123")
    assert {"qa-agent", "segment:start", "project:shop"} <= set(root["tags"])

    # Every run belongs to this trace and carries the QA run id.
    assert all(r.get("trace_id") == segment.trace_id for r in runs if "trace_id" in r)
    assert all(r["extra"]["metadata"]["qa_run_id"] == "run0001abcd" for r in runs if "extra" in r)

    names = {r.get("name") for r in runs}
    types = {r.get("run_type") for r in runs}
    assert {"requirement_analyzer", "test_executor", "root_cause_analyzer"} <= names  # LangGraph nodes
    assert "llm" in types  # LLM calls
    assert "knowledge_base_search" in names  # RAG retrieval
    assert {"api_test", "ui_test"} <= names  # tool calls
    [failure] = [r for r in runs if r.get("name") == "node_failure"]  # absorbed node error
    assert "confidence service unavailable" in failure["error"]
    assert outcome.status in ("awaiting_review", "completed")


def test_traced_tool_inputs_are_redacted():
    traces = CapturedTraces()

    @tracing.traced("probe_tool")
    def call_service(headers: dict, body: dict) -> dict:
        return {"echo": headers}

    with tracing_context(enabled=True, client=traces.client, project_name="qa-test"):
        call_service({"Authorization": "Bearer live-token-999"}, {"email": "a@b.c"})
    time.sleep(0.3)

    [run] = [r for r in traces.runs() if r.get("name") == "probe_tool"]
    assert "live-token-999" not in json.dumps(run)
    assert run["inputs"]["headers"]["Authorization"] == "***"
    assert run["inputs"]["body"] == {"email": "a@b.c"}


# --- correlation through the API -------------------------------------------------


def test_request_and_run_ids_flow_from_http_headers_to_background_logs(caplog):
    graph = build_qa_graph(
        {"test_executor": make_mock_test_executor({})},
        llm=FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
    )
    client = TestClient(create_app(lambda: RunManager(RunService(graph))))
    caplog.handler.addFilter(ContextFilter())
    caplog.set_level(logging.INFO)

    response = client.post(
        "/runs",
        json={"requirement": PASSWORD_RESET_REQUIREMENT, "project": "shop"},
        headers={"X-Request-ID": "ui-req-0001", "X-Client-Session": "browser-42"},
    )
    run_id = response.json()["run_id"]
    assert response.headers["X-Request-ID"] == "ui-req-0001"
    assert response.headers["X-QA-Run-ID"] == run_id

    deadline = time.monotonic() + 20
    while client.get(f"/runs/{run_id}").json()["status"] == "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    detail = client.get(f"/runs/{run_id}", headers={"X-Request-ID": "bad id with spaces"})
    assert len(detail.headers["X-Request-ID"]) == 32  # malformed ids are replaced
    assert detail.headers["X-QA-Run-ID"] == run_id
    assert [t["segment"] for t in detail.json()["traces"]] == ["start"]

    node_records = [r for r in caplog.records if r.getMessage().startswith("Node test_planner finished")]
    assert node_records, "node log line missing"
    ctx = node_records[0].qa_context
    assert (ctx["run_id"], ctx["request_id"], ctx["client_session"], ctx["node"], ctx["project"]) == (
        run_id, "ui-req-0001", "browser-42", "test_planner", "shop",
    )
    llm_lines = [r for r in caplog.records if r.getMessage().startswith("LLM call finished")]
    assert llm_lines and all(r.qa_context["run_id"] == run_id for r in llm_lines)


def test_database_writes_are_logged_with_the_run_id(db_sessions, caplog):
    from qa_agent.persistence.service import PersistenceService

    graph = build_qa_graph(
        {"test_executor": make_mock_test_executor({})},
        llm=FakeStructuredChatModel(responses=password_reset_responses()),
        knowledge_base=repo_knowledge_base(),
    )
    caplog.handler.addFilter(ContextFilter())
    caplog.set_level(logging.INFO, logger="qa_agent.persistence.service")

    outcome = RunService(graph, PersistenceService(db_sessions)).start(PASSWORD_RESET_REQUIREMENT, run_id="dbrun0001")

    db_lines = [r for r in caplog.records if r.getMessage().startswith("DB: agent_runs dbrun0001")]
    assert [r.getMessage().split()[3] for r in db_lines] == ["created", "synced"]
    assert "status=completed" in db_lines[-1].getMessage() and outcome.status == "completed"
