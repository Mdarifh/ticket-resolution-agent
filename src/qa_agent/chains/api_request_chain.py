"""API test cases (+ API docs from RAG) -> concrete HTTP requests.

Test cases describe steps in prose; this chain turns each API test case into
one ``ApiTestRequest``. The LLM only ever produces *relative paths*: the
target host always comes from configuration, so a model cannot redirect
traffic to an arbitrary server.
"""

import json
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import KnowledgeSearchResult, TestCase
from qa_agent.domain.api_test import ApiTestRequest, HttpMethod, ValidationRule, ValidationRuleType
from qa_agent.rag.context import REFERENCE_RULES, format_reference_material


class NameValue(BaseModel):
    name: str
    value: str


class ApiValidationRuleSpec(BaseModel):
    type: ValidationRuleType
    target: str | None = Field(
        description="JSON path such as 'data.items[0].id' for json_path_* rules, header name "
        "for header_* rules, null otherwise."
    )
    expected_json: str | None = Field(
        description="Expected value encoded as JSON (e.g. '\"token_used\"', '202', 'true'); "
        "for json_path_type one of \"string\", \"number\", \"integer\", \"boolean\", \"array\", "
        "\"object\", \"null\" (JSON-encoded); null for rules without an expected value."
    )


class ApiRequestSpec(BaseModel):
    test_case_id: str
    method: HttpMethod
    path: str = Field(description="Path relative to the API base URL, starting with '/'.")
    headers: list[NameValue] = Field(description="Extra headers; never include credentials.")
    query_params: list[NameValue]
    body_json: str | None = Field(description="Request body encoded as JSON, or null.")
    expected_status: int
    validation_rules: list[ApiValidationRuleSpec]
    skip_reason: str | None = Field(
        description="Why this test cannot be executed as a single request (e.g. it needs state "
        "that cannot be created, such as an expired token); null when executable."
    )


class ApiRequestPlan(BaseModel):
    requests: list[ApiRequestSpec]


API_REQUEST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You convert API test cases into concrete HTTP requests for an automated runner.\n"
            "Rules:\n"
            "- Produce exactly one request per test case id.\n"
            "- path is relative to the API base URL and starts with '/'. Never output a scheme, "
            "hostname or full URL; the runner decides where requests go.\n"
            "- Take endpoints, fields and status codes from the API reference material. If it "
            "does not document the endpoint, set skip_reason instead of guessing.\n"
            "- If the test needs prior state that a single request cannot create (an expired "
            "or already-used token, a delivered email), set skip_reason, unless the test case "
            "provides concrete data for that state.\n"
            "- Never include credentials or Authorization headers; authentication is added by "
            "the runner.\n"
            "- Add validation_rules for the observable expectations in the test case beyond "
            "the status code.\n\n" + REFERENCE_RULES,
        ),
        ("human", "API test cases (JSON):\n{test_cases}\n\n{reference_material}"),
    ]
)


def build_api_request_chain(llm: BaseChatModel) -> Runnable[dict, ApiRequestPlan]:
    return structured_chain(API_REQUEST_PROMPT, llm, ApiRequestPlan)


def generate_api_requests(
    llm: BaseChatModel, test_cases: list[TestCase], knowledge: KnowledgeSearchResult
) -> ApiRequestPlan:
    payload = "[" + ",".join(tc.model_dump_json() for tc in test_cases) + "]"
    return build_api_request_chain(llm).invoke(
        {"test_cases": payload, "reference_material": format_reference_material(knowledge)}
    )


class InvalidApiRequestSpec(ValueError):
    pass


def spec_to_request(spec: ApiRequestSpec) -> ApiTestRequest:
    """Validate an LLM request spec and convert it; raises ``InvalidApiRequestSpec``."""
    path = spec.path.strip()
    if not path.startswith("/") or path.startswith("//") or "://" in path:
        raise InvalidApiRequestSpec(f"Request path must be relative (start with '/'): {spec.path!r}")

    try:
        return ApiTestRequest(
            test_case_id=spec.test_case_id,
            method=spec.method,
            url=path,
            headers={h.name: h.value for h in spec.headers},
            query_params={q.name: q.value for q in spec.query_params},
            body=_parse_json(spec.body_json, "body_json"),
            expected_status=spec.expected_status,
            validation_rules=[
                ValidationRule(
                    type=rule.type,
                    target=rule.target,
                    expected=_parse_json(rule.expected_json, f"expected_json of {rule.type}"),
                )
                for rule in spec.validation_rules
            ],
        )
    except InvalidApiRequestSpec:
        raise
    except ValueError as exc:  # includes pydantic.ValidationError
        raise InvalidApiRequestSpec(f"Invalid request spec: {exc}") from exc


def _parse_json(value: str | None, field: str) -> Any:
    if value is None or value.strip() == "":
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise InvalidApiRequestSpec(f"{field} is not valid JSON: {exc.msg}") from exc
