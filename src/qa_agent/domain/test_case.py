"""Generated test cases and the suite that groups them.

These models are also LLM structured-output schemas. They avoid free-form
dicts (``test_data`` is a list of name/value pairs) so they stay compatible
with strict JSON-schema structured outputs.
"""

from typing import ClassVar, Literal

from pydantic import BaseModel, Field

TestType = Literal["positive", "negative", "edge_case"]
AutomationType = Literal["api", "ui", "manual"]
TestPriority = Literal["high", "medium", "low"]


class TestDataItem(BaseModel):
    __test__: ClassVar[bool] = False  # not a pytest test class

    name: str = Field(description="Input or fixture name, e.g. 'email'.")
    value: str = Field(description="Concrete value to use, e.g. 'registered.user@example.com'.")


class TestCase(BaseModel):
    __test__: ClassVar[bool] = False  # not a pytest test class

    test_case_id: str = Field(description="Identifier such as 'TC-001'.")
    title: str = Field(description="Short, specific name of what the test verifies.")
    description: str = Field(description="What behaviour the test covers and why it matters.")
    preconditions: list[str] = Field(description="State required before running the steps.")
    test_data: list[TestDataItem] = Field(description="Concrete inputs used by the steps.")
    steps: list[str] = Field(description="Ordered, atomic actions to perform.")
    expected_result: str = Field(description="Observable outcome that means the test passed.")
    priority: TestPriority
    test_type: TestType = Field(
        description="positive = valid use, negative = invalid input or misuse, "
        "edge_case = boundary or unusual conditions."
    )
    automation_type: AutomationType = Field(
        description="api = verifiable through HTTP requests/responses, ui = needs browser "
        "interaction, manual = needs human judgement or systems a script cannot reach."
    )


class TestSuite(BaseModel):
    __test__: ClassVar[bool] = False  # not a pytest test class

    name: str
    description: str
    test_cases: list[TestCase]
    knowledge_sources: list[str] = Field(
        description="Source paths from the reference material that informed these tests; "
        "empty if none did."
    )
