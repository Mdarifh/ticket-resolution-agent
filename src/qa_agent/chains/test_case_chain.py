"""``RequirementAnalysis`` (+ retrieved QA knowledge) -> ``TestSuite``."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import KnowledgeSearchResult, RequirementAnalysis, TestSuite
from qa_agent.rag.context import REFERENCE_RULES, format_reference_material

TEST_CASE_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior QA engineer. Design a test suite from the requirement analysis.\n"
            "Rules:\n"
            "- Cover every acceptance criterion.\n"
            "- Include positive, negative and edge_case tests; use the analysis's edge cases, "
            "business rules and risk areas.\n"
            "- Use concrete test data values, not placeholders.\n"
            "- Steps are ordered and atomic; the expected result is observable.\n"
            "- Suggest automation_type per test: api, ui or manual.\n"
            "- Do not resolve the analysis's ambiguities by inventing values; if a test "
            "depends on one, say so in its description.\n"
            "- Where the reference material documents relevant contracts, limits or past "
            "bugs, use them (e.g. add regression tests for past bugs) and name the source in "
            "the test description. Record every source you used in knowledge_sources.\n\n"
            + REFERENCE_RULES,
        ),
        (
            "human",
            "Requirement:\n{requirement}\n\n"
            "Requirement analysis (JSON):\n{analysis}\n\n"
            "{reference_material}",
        ),
    ]
)


def build_test_case_chain(llm: BaseChatModel) -> Runnable[dict, TestSuite]:
    return structured_chain(TEST_CASE_GENERATION_PROMPT, llm, TestSuite)


def generate_test_suite(
    llm: BaseChatModel,
    requirement_text: str,
    analysis: RequirementAnalysis,
    knowledge: KnowledgeSearchResult | None = None,
) -> TestSuite:
    knowledge = knowledge or KnowledgeSearchResult(query="", message="retrieval not performed")
    suite = build_test_case_chain(llm).invoke(
        {
            "requirement": requirement_text,
            "analysis": analysis.model_dump_json(indent=2),
            "reference_material": format_reference_material(knowledge),
        }
    )
    return normalize_test_ids(suite)


def normalize_test_ids(suite: TestSuite) -> TestSuite:
    """Renumber to TC-001.. so ids are unique and stable regardless of what the LLM chose."""
    renumbered = [
        tc.model_copy(update={"test_case_id": f"TC-{index:03d}"})
        for index, tc in enumerate(suite.test_cases, start=1)
    ]
    return suite.model_copy(update={"test_cases": renumbered})
