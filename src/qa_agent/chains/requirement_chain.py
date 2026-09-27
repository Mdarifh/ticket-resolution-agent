"""Requirement text -> ``RequirementAnalysis``."""

from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from qa_agent.chains._structured import structured_chain
from qa_agent.domain import Requirement, RequirementAnalysis

REQUIREMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a senior QA analyst. Analyse the software requirement and extract "
            "everything a tester needs to design tests.\n"
            "Rules:\n"
            "- Acceptance criteria must be atomic and individually testable.\n"
            "- Only state what the requirement says or clearly implies. Anything it leaves "
            "open (limits, timeouts, formats, policies) goes in ambiguities, not in rules.\n"
            "- Include security and abuse risks where relevant.",
        ),
        ("human", "Requirement:\n{requirement}"),
    ]
)


def build_requirement_analysis_chain(llm: BaseChatModel) -> Runnable[dict, RequirementAnalysis]:
    return structured_chain(REQUIREMENT_ANALYSIS_PROMPT, llm, RequirementAnalysis)


def analyze_requirement(llm: BaseChatModel, requirement: Requirement) -> RequirementAnalysis:
    return build_requirement_analysis_chain(llm).invoke({"requirement": requirement.text})
