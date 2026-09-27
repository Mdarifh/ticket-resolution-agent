# AI QA Agent

An agentic AI system that takes a software requirement and autonomously analyzes it, generates test cases, plans and executes API/UI tests, diagnoses failures with root-cause analysis backed by a RAG knowledge base, produces structured bug reports, scores its own confidence, escalates to a human when needed, and emits a final QA report.

This document is the **architecture plan only**. No application code has been written yet. It is meant to be reviewed and approved before implementation begins.

---

## 1. Goals & Non-Goals

**Goals**
- Demonstrate genuine, non-cosmetic use of LangChain (LLM calls, structured output, tools, retrievers) and LangGraph (stateful, conditional, cyclic workflow control with checkpointing and human-in-the-loop interrupts).
- Support both API testing and UI testing (Playwright) as first-class, independently invokable executor nodes.
- Show a real RAG loop: historical bugs/QA knowledge stored in a vector store, retrieved during root-cause analysis and bug report generation, and *written back to* after each run (so the knowledge base grows).
- Be runnable locally via Docker Compose, with a FastAPI backend, Streamlit frontend, and Postgres for structured persistence.
- Be portfolio-quality: clean layering, typed models, tested, documented, observable (LangSmith tracing).

**Non-goals (for v1)**
- Not a general-purpose test automation framework — it targets a single requirement → single test cycle per run.
- No multi-tenant auth/RBAC (single-user/local demo is fine; note extension points).
- No distributed execution (single worker process is enough for a portfolio project).

---

## 2. High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                              Streamlit UI                                │
│   - Submit requirement        - Watch live workflow status               │
│   - Approve/reject (HITL)     - View final QA report + bug reports       │
└───────────────────────────────┬────────────────────────────────────────--┘
                                 │ HTTP (REST + SSE/WebSocket for streaming)
┌───────────────────────────────▼────────────────────────────────────────--┐
│                              FastAPI Service                             │
│  ┌──────────────┐  ┌────────────────────┐  ┌───────────────────────────┐ │
│  │  API routers │  │  Run orchestrator  │  │  Auth/config (simple key) │ │
│  └──────┬───────┘  └─────────┬──────────┘  └───────────────────────────┘ │
│         │                    │                                          │
│         │           ┌────────▼─────────┐                                │
│         │           │  LangGraph Graph │  (compiled once, invoked/run)  │
│         │           │  + Checkpointer  │  (Postgres-backed)             │
│         │           └────────┬─────────┘                                │
│         │                    │  nodes call...                           │
│         │      ┌─────────────┼───────────────────────┐                  │
│         │      ▼             ▼                       ▼                  │
│         │  LangChain     Tools layer            RAG layer               │
│         │  (LLM chains,  (API test exec,        (Chroma/FAISS +         │
│         │   structured   Playwright exec,        embeddings, retriever, │
│         │   output,      pytest runner)          ingestion pipeline)    │
│         │   parsers)                                                    │
└─────────┼─────────────────────────────────────────────────────────────--┘
          │
┌─────────▼────────────────────────────┐   ┌─────────────────────────────┐
│           PostgreSQL                  │   │     Vector Store            │
│  runs, requirements, test_cases,      │   │  (Chroma persistent, local  │
│  test_results, bug_reports, hitl_     │   │   dir; FAISS as swappable   │
│  reviews, qa_knowledge_docs (mirror)  │   │   alt. behind one interface)│
└────────────────────────────────────---┘   └──────────────────────────--─┘

LangSmith: tracing/observability wraps every LangChain/LangGraph call (env-var opt-in).
```

**Key architectural decisions**
- **LangGraph owns control flow.** It is the only place that decides "what happens next." Nodes are thin wrappers that call LangChain chains/tools and return partial state updates.
- **Postgres-backed checkpointer** so a run that pauses for human approval can suspend and resume later (`interrupt_before`/`interrupt_after` in LangGraph), even across process restarts.
- **RAG is a real retriever tool**, not decoration: it's used by the Root Cause Analyzer and Bug Report Generator as a LangChain `Retriever` wrapped as a tool, and the pipeline writes new resolved-bug documents back into the store after each run, so the knowledge base compounds over time.
- **Ports & adapters for executors.** API test execution and Playwright UI execution are both implemented as LangChain `Tool`s with a common `ExecutionResult` contract, so the Test Executor node can dispatch to either (or both) based on the test plan without branching logic leaking into the graph.
- **Confidence + HITL is a graph-level concern**, implemented as a conditional edge, not buried in a node — so it's visible in the workflow diagram and easy to reason about/demo.

---

## 3. Folder Structure

```
ai-qa-agent/
├── README.md                        # this planning doc (kept updated as build reference)
├── pyproject.toml                   # single Python project, managed with uv/poetry
├── docker-compose.yml               # postgres + backend + streamlit + (optional) chroma server
├── Dockerfile                       # targets: api, ui, demo
├── .env.example
├── alembic.ini
├── migrations/                      # alembic migrations for Postgres schema
│   └── versions/
│
├── src/
│   └── qa_agent/
│       ├── __init__.py
│       ├── config.py                 # pydantic-settings, env-driven config
│       ├── logging_conf.py           # structured logging setup
│       │
│       ├── domain/                   # pure data models, no framework deps
│       │   ├── __init__.py
│       │   ├── requirement.py        # Requirement, RequirementAnalysis
│       │   ├── test_case.py          # TestCase, TestStep, TestType enum
│       │   ├── test_plan.py          # TestPlan, ExecutionOrder
│       │   ├── execution.py          # ExecutionResult, FailureDetail
│       │   ├── analysis.py           # FailureAnalysis, RootCause
│       │   ├── bug_report.py         # BugReport, Severity, Priority
│       │   ├── confidence.py         # ConfidenceScore, ConfidenceFactors
│       │   └── qa_report.py          # FinalQAReport
│       │
│       ├── graph/                    # LangGraph workflow definition
│       │   ├── __init__.py
│       │   ├── state.py              # QAWorkflowState (TypedDict/Pydantic)
│       │   ├── build_graph.py        # StateGraph wiring, edges, conditionals
│       │   ├── checkpointer.py       # Postgres checkpointer setup
│       │   └── nodes/
│       │       ├── __init__.py
│       │       ├── requirement_analyzer.py
│       │       ├── test_case_generator.py
│       │       ├── test_planner.py
│       │       ├── test_executor.py
│       │       ├── result_analyzer.py
│       │       ├── failure_analyzer.py
│       │       ├── root_cause_analyzer.py
│       │       ├── bug_report_generator.py
│       │       ├── confidence_checker.py
│       │       ├── human_review.py
│       │       └── final_report_generator.py
│       │
│       ├── chains/                   # LangChain: prompts + structured output chains
│       │   ├── __init__.py
│       │   ├── llm_provider.py       # ChatOpenAI (or compatible) factory, model config
│       │   ├── requirement_chain.py
│       │   ├── test_case_chain.py
│       │   ├── test_plan_chain.py
│       │   ├── failure_analysis_chain.py
│       │   ├── root_cause_chain.py
│       │   ├── bug_report_chain.py
│       │   ├── confidence_chain.py
│       │   └── final_report_chain.py
│       │
│       ├── tools/                    # LangChain Tools (executable actions)
│       │   ├── __init__.py
│       │   ├── base.py               # ExecutionResult contract, tool error handling
│       │   ├── api_test_tool.py      # httpx-based API test executor
│       │   ├── ui_test_tool.py       # Playwright-based UI test executor
│       │   ├── pytest_runner_tool.py # wraps generated pytest files, runs, parses results
│       │   └── rag_search_tool.py    # retriever-as-tool for historical QA knowledge
│       │
│       ├── rag/                      # RAG subsystem
│       │   ├── __init__.py
│       │   ├── embeddings.py         # embedding model factory
│       │   ├── vector_store.py       # Chroma/FAISS behind a common interface
│       │   ├── ingestion.py          # chunk + embed + upsert historical docs
│       │   ├── retriever.py          # LangChain retriever config (k, filters, MMR)
│       │   └── seed_data/            # sample historical bugs/QA notes for demo
│       │       └── *.md / *.json
│       │
│       ├── playwright_tests/         # generated & template Playwright specs
│       │   ├── conftest.py
│       │   └── templates/
│       │
│       ├── persistence/              # Postgres access layer
│       │   ├── __init__.py
│       │   ├── db.py                 # SQLAlchemy engine/session
│       │   ├── models.py             # ORM models (mirrors DB schema, §8)
│       │   └── repository.py         # CRUD/query functions used by API + graph nodes
│       │
│       ├── api/                      # FastAPI application
│       │   ├── __init__.py
│       │   ├── main.py               # app factory, middleware, startup/shutdown
│       │   ├── deps.py               # dependency-injected DB session, graph instance
│       │   ├── schemas.py            # pydantic request/response models
│       │   └── routers/
│       │       ├── runs.py           # POST /runs, GET /runs/{id}, GET /runs/{id}/stream
│       │       ├── reviews.py        # GET/POST /runs/{id}/review (HITL)
│       │       ├── reports.py        # GET /runs/{id}/report, /runs/{id}/bugs
│       │       └── knowledge.py      # RAG ingest/search endpoints (admin/demo use)
│       │
│       └── ui/                       # Streamlit frontend
│           ├── Home.py
│           └── pages/
│               ├── 1_Submit_Requirement.py
│               ├── 2_Run_Monitor.py
│               ├── 3_Human_Review.py
│               ├── 4_Bug_Reports.py
│               └── 5_QA_Report.py
│
├── tests/                            # project's own test suite (pytest)
│   ├── unit/
│   │   ├── test_chains.py
│   │   ├── test_nodes.py
│   │   ├── test_confidence.py
│   │   └── test_rag.py
│   ├── integration/
│   │   ├── test_graph_end_to_end.py
│   │   ├── test_api_routes.py
│   │   └── test_playwright_tool.py
│   └── fixtures/
│       └── sample_requirements.json
│
└── docs/
    ├── architecture.md               # (this content, split out post-approval)
    ├── workflow_diagram.md           # mermaid diagram of the LangGraph
    └── adr/                          # architecture decision records
        └── 0001-langgraph-for-control-flow.md
```

**Why this shape**
- `domain/` has zero LangChain/LangGraph imports — keeps business objects testable and framework-agnostic.
- `chains/` is where LangChain is "genuinely used": every chain here does an LLM call with a bound structured-output schema (via `.with_structured_output()` or a Pydantic output parser), not a raw string prompt.
- `graph/nodes/` are thin: parse input state → call a chain/tool → shape output state. No business logic duplicated between nodes and chains.
- `tools/` is where side effects live (HTTP calls, browser automation, pytest subprocess) — isolated so they can be mocked in tests.
- One Python package (`qa_agent`), imported by both FastAPI and Streamlit, avoids duplicating domain logic across services.

---

## 4. LangGraph State

```python
class QAWorkflowState(TypedDict, total=False):
    # input
    run_id: str
    requirement_text: str
    requirement_metadata: dict  # e.g. {"app": "checkout", "priority": "high"}

    # requirement analysis
    requirement_analysis: RequirementAnalysis | None

    # test design
    test_cases: list[TestCase]
    test_plan: TestPlan | None

    # execution
    execution_results: list[ExecutionResult]
    failed_results: list[ExecutionResult]

    # analysis
    failure_analyses: list[FailureAnalysis]
    root_causes: list[RootCause]
    retrieved_knowledge: list[RetrievedDoc]      # RAG hits used for RCA/bug report

    # output artifacts
    bug_reports: list[BugReport]
    confidence: ConfidenceScore | None

    # human-in-the-loop
    requires_human_review: bool
    review_reason: str | None
    human_decision: HumanDecision | None          # approve / reject / edit
    human_feedback: str | None

    # final
    final_report: FinalQAReport | None

    # control/meta
    status: Literal["running", "paused_for_review", "completed", "failed"]
    errors: list[WorkflowError]                   # accumulated node-level errors
    retry_counts: dict[str, int]                  # per-node retry tracking
```

Design notes:
- State is a **typed accumulator**, updated via LangGraph's reducer pattern (e.g. `Annotated[list[X], operator.add]` for lists that grow, plain overwrite for scalars).
- `status` + `errors` make failure states first-class and inspectable from the API/UI without inferring from exceptions.
- `human_decision`/`human_feedback` are populated by the API's review endpoint and fed back into the graph on resume — this is what makes HITL a real interrupt/resume, not a polling hack.

---

## 5. LangGraph Workflow

```
                         ┌────────────────────┐
                         │ Requirement Analyzer│
                         └──────────┬─────────┘
                                    ▼
                         ┌────────────────────┐
                         │ Test Case Generator │
                         └──────────┬─────────┘
                                    ▼
                         ┌────────────────────┐
                         │    Test Planner     │
                         └──────────┬─────────┘
                                    ▼
                         ┌────────────────────┐
                         │   Test Executor     │◄──────┐ (dispatches per-test to
                         │ (API + UI/Playwright)│       │  api_test_tool / ui_test_tool)
                         └──────────┬─────────┘        │
                                    ▼                   │
                         ┌────────────────────┐         │
                         │  Result Analyzer    │         │
                         └──────────┬─────────┘         │
                            any failures? ── no ───────────────┐
                                    │ yes                        │
                                    ▼                             │
                         ┌────────────────────┐                  │
                         │  Failure Analyzer   │                  │
                         └──────────┬─────────┘                  │
                                    ▼                             │
                         ┌────────────────────┐                  │
                         │ Root Cause Analyzer │◄── RAG retriever │
                         │ (+ historical search)│                 │
                         └──────────┬─────────┘                  │
                                    ▼                             │
                         ┌────────────────────┐                  │
                         │ Bug Report Generator│                  │
                         └──────────┬─────────┘                  │
                                    ▼                             │
                         ┌────────────────────┐                  │
                         │ Confidence Checker  │                  │
                         └──────────┬─────────┘                  │
                     low confidence │ or sensitive action         │
                            ┌───────┴────────┐                    │
                            │ yes            │ no                 │
                            ▼                │                    │
                 ┌────────────────────┐      │                    │
                 │   Human Review      │      │                    │
                 │  (graph interrupt)  │      │                    │
                 └──────────┬─────────┘      │                    │
                   approved/edited│           │                    │
                            └───────┬────────┘                    │
                                    ▼                             │
                         ┌────────────────────┐                  │
                         │ Final Report Gen.   ◄───────────────────┘
                         └──────────┬─────────┘
                                    ▼
                                  [END]
```

**Conditional edges**
1. `Result Analyzer → {Failure Analyzer | Final Report Generator}` based on whether any `execution_results` failed.
2. `Confidence Checker → {Human Review | Final Report Generator}` based on `confidence.score < threshold OR action is sensitive` (sensitive = e.g. bug auto-filed to external tracker, or root cause implicates a specific team/person).
3. `Human Review → {Bug Report Generator (regenerate) | Final Report Generator | END(rejected)}` based on `human_decision`:
   - `approve` → proceed to Final Report Generator.
   - `request_changes` → loop back to Bug Report Generator (or Root Cause Analyzer if the human flags RCA itself as wrong) with `human_feedback` injected into the chain's prompt context.
   - `reject` → mark run failed/cancelled, still produce a minimal final report noting rejection.

**Interrupt mechanism:** `Human Review` node is registered with LangGraph's `interrupt_before` (or the `interrupt()` primitive in newer LangGraph) so the graph execution suspends, persists state via the Postgres checkpointer, and returns control to the API. The API exposes a `POST /runs/{id}/review` endpoint that writes the decision into state and calls `graph.invoke(None, config={"configurable": {"thread_id": run_id}})` to resume.

**Retry/self-loop:** `Test Executor` may retry a single flaky test up to N times (tracked via `retry_counts`) before treating it as a genuine failure — implemented as a node-internal loop, not a graph edge, to keep the graph diagram clean.

---

## 6. Node (Agent) Responsibilities

| Node | Responsibility | LangChain usage | Reads (state) | Writes (state) |
|---|---|---|---|---|
| Requirement Analyzer | Parse free-text requirement into structured intent: actors, preconditions, acceptance criteria, affected components, risk areas | Structured-output chain (Pydantic schema) | `requirement_text` | `requirement_analysis` |
| Test Case Generator | Produce positive/negative/edge/boundary test cases per acceptance criterion, tagged API vs UI | Structured-output chain, few-shot examples | `requirement_analysis` | `test_cases` |
| Test Planner | Order test cases (dependency-aware), assign executor type, decide parallelizable groups, mark data setup needs | Structured-output chain + deterministic post-processing | `test_cases` | `test_plan` |
| Test Executor | Run each planned test via the correct tool (API or Playwright), capture raw results | Tool-calling (LangChain Tools), no LLM needed for the execution itself | `test_plan` | `execution_results` |
| Result Analyzer | Classify pass/fail, compute summary stats, decide branch | Lightweight — mostly deterministic; LLM used only to normalize ambiguous assertions | `execution_results` | `failed_results`, routing signal |
| Failure Analyzer | For each failure: categorize (assertion mismatch, timeout, 5xx, element not found, flaky, env issue) and extract signal (stack trace, response diff, screenshot/DOM snapshot) | Structured-output chain | `failed_results` | `failure_analyses` |
| Root Cause Analyzer | Correlate failure signals with historical incidents via RAG; hypothesize root cause with confidence per hypothesis | RAG retriever tool + reasoning chain | `failure_analyses`, `retrieved_knowledge` | `root_causes`, `retrieved_knowledge` |
| Bug Report Generator | Compose structured bug report (title, repro steps, severity, priority, evidence, linked root cause) | Structured-output chain | `root_causes`, `failure_analyses` | `bug_reports` |
| Confidence Checker | Score overall confidence (test coverage adequacy, RCA certainty, evidence strength) and flag sensitivity | Structured-output chain + deterministic rules (see §confidence) | `bug_reports`, `root_causes`, `test_plan` | `confidence`, `requires_human_review`, `review_reason` |
| Human Review | Surface run to a human, suspend graph, resume with decision | No LLM call itself (pure interrupt); optional LLM to summarize "what needs review" | `confidence`, `bug_reports` | `human_decision`, `human_feedback`, `status` |
| Final Report Generator | Aggregate everything into a polished QA report (executive summary, coverage, results, bugs, RCA, confidence, recommendations) | Structured-output chain + template rendering (Markdown/HTML) | full state | `final_report` |

---

## 7. Tools (LangChain `Tool`/`StructuredTool`)

| Tool | Purpose | Backing tech | Notes |
|---|---|---|---|
| `api_test_tool` | Executes a single API test case (method, url, headers, body, assertions) | `httpx` | Returns `ExecutionResult` (status, latency, assertion pass/fail, raw response) |
| `ui_test_tool` | Executes a single UI test case against a target app | Playwright (async API) | Generates/loads a Playwright script from the test case spec; captures screenshot + trace on failure |
| `pytest_runner_tool` | Runs a batch of generated pytest files (used for API tests expressed as pytest, and to standardize result parsing) | `pytest` + `pytest-json-report` | Parses JSON report into `ExecutionResult[]` |
| `rag_search_tool` | Semantic search over historical QA knowledge base | LangChain retriever wrapping Chroma/FAISS | Used by Root Cause Analyzer and optionally Bug Report Generator |
| `bug_tracker_tool` *(stub/optional, marked sensitive)* | Would file a bug in an external tracker (Jira-like) | HTTP client, mocked in v1 | Marked as a "sensitive action" — its use always forces `requires_human_review = true` |

All tools share a common error-handling contract (`base.py`): timeouts, retries with backoff for transient network errors, and a normalized `ToolError` that nodes catch and record into `state.errors` rather than crashing the graph.

---

## 8. RAG Architecture

**Purpose:** give Root Cause Analysis and Bug Report Generation access to prior QA history — past bugs, resolutions, known flaky areas, incident postmortems — so the agent reasons with institutional memory instead of a blank slate.

**Corpus (seeded + growing)**
- Seed data (`rag/seed_data/`): a curated set of ~30-50 synthetic-but-realistic historical bug reports/postmortems for the demo app, in Markdown/JSON, covering common categories (auth failures, race conditions, validation gaps, UI flakiness, integration timeouts).
- Growing data: after each completed run, the Final Report Generator (or a dedicated post-run hook) writes new `bug_reports` + their confirmed `root_causes` back into the vector store as new documents — so the RAG store compounds across runs, demonstrating a real feedback loop.

**Pipeline**
1. **Ingestion** (`rag/ingestion.py`): load documents → chunk (semantic/markdown-aware splitter, ~500 token chunks with overlap) → embed → upsert into vector store with metadata (`doc_type`, `component`, `date`, `severity`, `source_run_id`).
2. **Storage**: `rag/vector_store.py` defines a `VectorStoreBackend` protocol implemented by both a `ChromaBackend` (default, persistent local dir, zero extra infra) and a `FAISSBackend` (alt, for demonstrating swappability) — selected via config (`VECTOR_STORE=chroma|faiss`).
3. **Retrieval**: `rag/retriever.py` builds a LangChain retriever (`as_retriever`) with MMR search (diversity) and optional metadata filters (e.g. same `component` as the failing test). Exposed as a `Tool` for the Root Cause node, and also usable directly as a LangChain retriever inside an LCEL chain for the Bug Report Generator ("cite similar past bugs").
4. **Grounding**: retrieved chunks are injected into the RCA/bug-report chain prompt as labeled context with source metadata, and the chain is instructed to cite which historical doc (if any) informed its hypothesis — surfaced in the final bug report as "related past incidents."

**Why Chroma as default:** zero-ops local persistence (fits Docker Compose easily), good LangChain support, easy to swap for FAISS or a hosted store later — demonstrates the adapter pattern without adding infra weight to a portfolio project.

---

## 9. Database Schema (PostgreSQL)

Postgres is the system of record for runs and their artifacts (audit trail, HITL history, reporting) — separate concern from the vector store, which holds unstructured semantic knowledge.

```sql
-- one row per requirement submitted
CREATE TABLE requirements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- one row per workflow execution (maps 1:1 to a LangGraph thread_id)
CREATE TABLE runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    requirement_id UUID NOT NULL REFERENCES requirements(id),
    status TEXT NOT NULL CHECK (status IN
        ('running','paused_for_review','completed','failed','rejected')),
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    confidence_score NUMERIC(4,3),
    requires_human_review BOOLEAN NOT NULL DEFAULT false,
    review_reason TEXT
);

CREATE TABLE requirement_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    acceptance_criteria JSONB NOT NULL,
    actors JSONB,
    risk_areas JSONB,
    raw_llm_output JSONB
);

CREATE TABLE test_cases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    title TEXT NOT NULL,
    test_type TEXT NOT NULL CHECK (test_type IN ('api','ui')),
    category TEXT,              -- positive/negative/edge/boundary
    steps JSONB NOT NULL,
    expected_result TEXT,
    priority TEXT
);

CREATE TABLE test_plan_entries (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    test_case_id UUID NOT NULL REFERENCES test_cases(id),
    execution_order INT NOT NULL,
    parallel_group INT,
    executor_type TEXT NOT NULL CHECK (executor_type IN ('api','playwright'))
);

CREATE TABLE test_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    test_case_id UUID NOT NULL REFERENCES test_cases(id),
    status TEXT NOT NULL CHECK (status IN ('passed','failed','error','skipped')),
    duration_ms INT,
    raw_output JSONB,           -- response body / DOM snapshot / trace ref
    screenshot_path TEXT,
    executed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE failure_analyses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    test_result_id UUID NOT NULL REFERENCES test_results(id),
    failure_category TEXT,      -- assertion_mismatch/timeout/5xx/element_not_found/flaky/env
    signal JSONB
);

CREATE TABLE root_causes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    failure_analysis_id UUID NOT NULL REFERENCES failure_analyses(id),
    hypothesis TEXT NOT NULL,
    confidence NUMERIC(4,3),
    supporting_evidence JSONB,
    related_knowledge_doc_ids JSONB  -- vector store doc ids cited
);

CREATE TABLE bug_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    root_cause_id UUID REFERENCES root_causes(id),
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    repro_steps JSONB NOT NULL,
    severity TEXT CHECK (severity IN ('low','medium','high','critical')),
    priority TEXT CHECK (priority IN ('p0','p1','p2','p3')),
    evidence JSONB,
    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft','approved','rejected','filed')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE hitl_reviews (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL REFERENCES runs(id),
    reason TEXT NOT NULL,
    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewer TEXT,
    decision TEXT CHECK (decision IN ('approve','request_changes','reject')),
    feedback TEXT,
    decided_at TIMESTAMPTZ
);

CREATE TABLE final_reports (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id UUID NOT NULL UNIQUE REFERENCES runs(id),
    summary TEXT NOT NULL,
    report_markdown TEXT NOT NULL,
    metrics JSONB,               -- pass rate, coverage, bug count by severity, etc.
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- LangGraph's own checkpointer will manage its persistence tables
-- (via langgraph-checkpoint-postgres); listed here only for awareness,
-- not hand-authored.
```

Managed with **Alembic** migrations from day one (even for a portfolio project — shows real engineering hygiene).

---

## 10. FastAPI Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/requirements` | Submit a new requirement (creates `requirements` row) |
| `POST` | `/runs` | Start a new QA workflow run for a requirement (kicks off LangGraph invocation, async) |
| `GET` | `/runs/{run_id}` | Get current run status + state snapshot |
| `GET` | `/runs/{run_id}/stream` | Stream live node-by-node progress (SSE) for the UI |
| `GET` | `/runs/{run_id}/test-cases` | List generated test cases |
| `GET` | `/runs/{run_id}/results` | List execution results |
| `GET` | `/runs/{run_id}/bugs` | List generated bug reports |
| `GET` | `/runs/{run_id}/review` | Get pending review details (reason, confidence, artifacts to review) |
| `POST` | `/runs/{run_id}/review` | Submit human decision (`approve`/`request_changes`/`reject` + feedback) → resumes graph |
| `GET` | `/runs/{run_id}/report` | Get final QA report (Markdown/JSON) |
| `GET` | `/runs` | List runs (filter by status, date) |
| `POST` | `/knowledge/ingest` | Manually ingest a document into the RAG store (admin/demo) |
| `GET` | `/knowledge/search` | Debug endpoint: run a raw similarity search |
| `GET` | `/health` | Liveness/readiness probe |

All request/response bodies are Pydantic schemas in `api/schemas.py`, separate from the internal `domain/` models (schema translation happens at the router boundary — keeps API contracts decoupled from internal refactors).

---

## 11. Streamlit Screens

1. **Submit Requirement** — textarea + metadata form → `POST /requirements` + `POST /runs`; shows resulting `run_id`.
2. **Run Monitor** — polls/streams `/runs/{id}/stream`; renders a live node-progress tracker (mirrors the LangGraph diagram) with per-node status (pending/running/done/failed) and elapsed time.
3. **Human Review** — appears when a run is `paused_for_review`; shows confidence breakdown, draft bug reports, root cause hypotheses, and an approve / request-changes / reject form with a feedback textbox → `POST /runs/{id}/review`.
4. **Bug Reports** — table + detail view of generated bug reports for a run (severity/priority filters, linked root cause, linked evidence/screenshots).
5. **QA Report** — final rendered Markdown/HTML report with summary metrics (pass rate, bug counts by severity, confidence, coverage), downloadable.

---

## 12. Configuration

Centralized in `config.py` via `pydantic-settings`, loaded from `.env`:

```
# LLM
OPENAI_API_KEY=
OPENAI_BASE_URL=              # empty = official OpenAI; else any OpenAI-compatible endpoint
LLM_MODEL=gpt-4o-mini          # swappable
LLM_TEMPERATURE=0.1

# LangSmith (see §17; the legacy LANGCHAIN_TRACING_V2 / LANGCHAIN_API_KEY names also work)
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=ai-qa-agent

# Vector store
VECTOR_STORE=chroma            # chroma | faiss
CHROMA_PERSIST_DIR=./data/chroma
EMBEDDING_MODEL=text-embedding-3-small

# Postgres
DATABASE_URL=postgresql+psycopg://qa:qa@postgres:5432/qa_agent

# Execution
API_TEST_BASE_URL=             # target system under test, API
UI_TEST_BASE_URL=              # target system under test, UI
PLAYWRIGHT_HEADLESS=true
TEST_EXECUTION_TIMEOUT_S=30

# Confidence / HITL
CONFIDENCE_THRESHOLD=0.7
SENSITIVE_ACTIONS=file_external_bug,mark_critical_severity

# App
APP_ENV=local
LOG_LEVEL=INFO
```

`.env.example` checked in; `.env` gitignored. Docker Compose passes these through; Streamlit and FastAPI both read the same `config.py` singleton (shared package, no duplication).

---

## 13. Testing Strategy

| Layer | What's tested | How |
|---|---|---|
| Domain models | Validation rules, enum constraints | Plain pytest, no mocks |
| Chains | Prompt → structured output shape correctness | pytest + recorded/mocked LLM responses (VCR-style cassette or a fake `ChatModel`); no live API calls in CI |
| Tools | `api_test_tool` against a local mock server (e.g. `httpx` + `respx`); `ui_test_tool` against a tiny bundled static HTML page via Playwright | pytest, marked `@pytest.mark.integration` where a browser/subprocess is needed |
| Graph nodes | Each node function in isolation: given a state slice, correct partial-state output | pytest, chain/tool calls mocked |
| Graph (end-to-end) | Full happy path (no failures → straight to report) and failure path (with HITL interrupt/resume) | pytest with a fake LLM + fake tools, asserting on final state and node visitation order |
| API | Route contracts, HITL resume flow, error responses | `httpx.AsyncClient` against the FastAPI app, test DB (dockerized Postgres or SQLite for speed where feasible) |
| RAG | Ingestion → retrieval returns expected top-k for known queries | pytest against a throwaway Chroma collection |
| Confidence logic | Threshold boundaries, sensitive-action flagging | Pure unit tests (deterministic rules table) |

CI (GitHub Actions, if repo is pushed to GitHub): lint (ruff) + type-check (mypy) + unit tests on every push; integration/E2E tests in a separate job with docker-compose services spun up.

---

## 14. Error Handling Strategy

- **Tool-level:** every tool call wrapped in try/except → normalized `ToolError(kind, message, retriable)`. Retriable errors (timeouts, transient 5xx) retried with exponential backoff (max 2 retries) before surfacing.
- **Node-level:** nodes catch chain/tool exceptions, append to `state["errors"]`, and set a node-specific fallback (e.g. if Test Case Generator fails, produce zero test cases and mark run `failed` rather than crashing the process).
- **Graph-level:** LangGraph run wrapped with a top-level handler in the orchestrator; unrecoverable errors mark `runs.status = 'failed'` in Postgres with the error persisted, so the UI can show a clear failure state instead of hanging.
- **LLM output validation:** every structured-output chain validates against its Pydantic schema; on validation failure, one automatic re-ask ("your output didn't match the schema, here's the error, retry") before giving up and recording a node error.
- **Playwright-specific:** browser/context always closed in a `finally`; on any UI test exception, capture screenshot + trace before re-raising as a `ToolError` so failures are debuggable.
- **Idempotency:** resuming a run after a human review always re-enters the graph via the checkpointer's saved state — nodes upstream of Human Review are not re-executed.

---

## 15. Human-in-the-Loop Design

**Triggers for review** (evaluated in Confidence Checker):
1. `confidence.overall_score < CONFIDENCE_THRESHOLD`.
2. Any bug report has `severity in (high, critical)`.
3. Root cause hypothesis confidence is itself low/ambiguous (multiple competing hypotheses with close scores).
4. A "sensitive action" flag is set (e.g. would auto-file to an external tracker, or the RCA implicates a specific external dependency/team).

**Mechanics**
- Implemented with LangGraph's native interrupt support: the graph pauses *before* `Human Review`, persists state to the Postgres checkpointer keyed by `thread_id = run_id`, and returns control to the caller (the FastAPI orchestrator).
- `runs.status = 'paused_for_review'`; API surfaces this immediately.
- Streamlit's Human Review screen polls/streams for this status and renders: why review was triggered (`review_reason`), the confidence breakdown, the draft bug report(s), and root cause reasoning with cited RAG sources.
- Reviewer submits one of three decisions:
  - **Approve** → resume graph, proceed to Final Report Generator as-is.
  - **Request changes** (with free-text feedback) → resume graph, route back to Bug Report Generator (or Root Cause Analyzer, if feedback indicates the RCA itself is wrong) with `human_feedback` injected into that chain's next prompt.
  - **Reject** → resume graph into a terminal "rejected" path; Final Report Generator still runs but produces a short report documenting the rejection and reason, run marked `rejected`.
- All review actions are audit-logged in `hitl_reviews` (who, when, decision, feedback) — supports a "why did the agent do X" narrative, which matters for a portfolio demo of responsible agentic design.

---

## 16. Implementation Phases

Building incrementally, each phase independently demoable:

1. **Phase 0 — Scaffolding** — repo structure, `pyproject.toml`, config, Docker Compose skeleton (Postgres only), Alembic init, empty FastAPI app with `/health`.
2. **Phase 1 — Domain + Chains (no graph yet)** — domain models; `llm_provider.py`; Requirement Analyzer and Test Case Generator chains with structured output, unit-tested against a fake LLM. Provable via a small script/CLI, not yet wired to LangGraph.
3. **Phase 2 — LangGraph skeleton** — `state.py`, `build_graph.py` wiring the first three nodes (Requirement Analyzer → Test Case Generator → Test Planner) linearly, Postgres checkpointer configured. Demo: run a requirement through to a test plan.
4. **Phase 3 — Execution layer** — `api_test_tool`, `ui_test_tool` (Playwright), `pytest_runner_tool`; Test Executor + Result Analyzer nodes; conditional edge to failure path vs. straight-through. Demo: real API test against a small sample target service; real Playwright run against a sample page.
5. **Phase 4 — RAG subsystem** — vector store backend(s), ingestion of seed historical data, retriever, `rag_search_tool`; Failure Analyzer, Root Cause Analyzer nodes wired in with retrieval grounding.
6. **Phase 5 — Bug reporting + confidence + HITL** — Bug Report Generator, Confidence Checker, Human Review interrupt/resume, Final Report Generator. Demo: full end-to-end run including a forced low-confidence path that pauses for review.
7. **Phase 6 — FastAPI surface** — all routers, SSE streaming for run progress, review endpoint wired to graph resume.
8. **Phase 7 — Streamlit UI** — all five screens against the real API.
9. **Phase 8 — Observability + polish** — LangSmith tracing verified end-to-end, structured logging, error-path tests, README usage docs, Docker Compose full stack (`docker compose up` brings up everything), CI pipeline.
10. **Phase 9 — RAG feedback loop closure** — write-back of completed run bug reports/root causes into the vector store, demonstrated by a second, related run producing better-grounded RCA citing the first run's bug.

Each phase lands as its own reviewable set of changes rather than one monolithic implementation — consistent with building this incrementally rather than all at once.

---

## 17. Observability

The system has two observability layers:
- **LangSmith traces**: a step-by-step record of what the agent did.
- **Application logs**: who called what, when, and how long it took.

The QA **run id** ties the two together. Code lives in `src/qa_agent/observability/` and `src/qa_agent/logging_conf.py`.

### LangSmith setup

1. Create an account at <https://smith.langchain.com>. Then go to **Settings → API Keys → Create API Key** and create a personal or service key (it starts with `lsv2_`).
2. Add the key to `.env`:

   ```
   LANGSMITH_TRACING=true
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_PROJECT=ai-qa-agent        # created automatically on the first trace
   # LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com   # EU region / self-hosted only
   ```

3. Restart the backend (`uvicorn qa_agent.api.main:app --app-dir src --port 8000`). The startup log should show `LangSmith tracing enabled (project=ai-qa-agent)`, and **System status** in the dashboard sidebar reports `langsmith_tracing: true`.
4. Start a run from the dashboard. Open **Test Run Status → 🔭 Observability** to see the run's traces, with **Open trace** links into LangSmith.

Notes:
- Tracing stays off unless *both* `LANGSMITH_TRACING=true` and a key are set. If the flag is on but the key is missing, the backend logs a warning, turns tracing off everywhere, and shows a configuration note in the UI. It never fails with 401s.
- `pydantic-settings` reads `.env`, but LangChain reads only real environment variables. So at startup `configure_tracing()` exports the `LANGSMITH_*` values into the process environment.
- **Privacy:** traces contain prompts, requirement text, generated tests, HTTP request/response bodies of the system under test, and knowledge base excerpts. Point tracing at a LangSmith workspace that is allowed to hold that data. Credentials are redacted (see below); test data is not.

### What is traced

Each **workflow segment** of a run is one LangSmith trace:
- `qa_run.start`: from the requirement to test planning, or to the end if the approval gate before execution is off.
- `qa_run.execute`: the tests, analysis and bug reports that follow approval.
- `qa_run.review`: everything after a human decision.

All segments of a run share `thread_id = run_id`, so LangSmith's **Threads** view groups them.

| What | How it appears in the trace |
|---|---|
| LangGraph execution | Root run `qa_run.<segment>` with one child run per node (`requirement_analyzer`, `test_executor`, ...) |
| LangChain operations | Prompt → structured-output chains, including the automatic schema-validation retry |
| LLM calls | `llm` runs with model, messages, output and token usage |
| Tool calls | `api_test` and `ui_test` spans (and `ui_page_inventory`) under `test_executor`, with inputs/outputs redacted |
| RAG retrieval | `knowledge_base_search` retriever spans: query, filters, returned chunks with scores and warnings |
| Failures | Exceptions mark their span as errored. Node errors that the workflow absorbs, and keeps running after, appear as an errored `node_failure` span. A crash of the whole segment errors the root run. |

Every run in a trace carries this metadata:
- `qa_run_id`
- `thread_id`
- `segment`
- `project`
- `request_id` (the HTTP request that started the segment)
- `client_session` (the Streamlit browser session)
- `app_env`

Every run is also tagged `qa-agent`, `segment:<name>` and `project:<name>`. To find every trace of a run in LangSmith, filter on metadata `qa_run_id = <run id>`.

### Following one QA run end to end

| Layer | Identifier | Where to look |
|---|---|---|
| Streamlit | `X-Client-Session` (one per browser session) and a fresh `X-Request-ID` per API call; `X-QA-Run-ID` on run endpoints | Streamlit log (`qa_agent.ui.api_client`). Error messages in the UI show `Reference: request <id>`. |
| FastAPI | `request_id` (taken from the header, or generated), `run_id` (from the path) | `qa_agent.api.access` log lines. The response headers `X-Request-ID` and `X-QA-Run-ID` echo them. |
| Background worker | The same `request_id`/`client_session` (context is copied into the worker thread), plus `run_id`, `project`, `segment` | `Segment start started/finished ...` log lines |
| LangGraph | `thread_id = run_id`; each segment's root run id is the **trace id** | LangSmith, `GET /runs/{id}` → `traces`, UI Observability panel |
| LangChain / LLM | Inherits the segment's metadata and callbacks | LangSmith `llm` runs; `LLM call finished model=... duration_ms=... tokens` log lines |
| Tools | Inherit the metadata; also bound as `node=test_executor` | LangSmith tool spans; `API test TC-001 POST https://... -> 202 passed (84 ms)` log lines |
| Database | `agent_runs.id = run_id` (every child table references it) | `DB: agent_runs <id> synced status=...` log lines |

Every log line prints the ids bound at that moment:

```
2026-09-27 00:12:03 | INFO    | run=5455144e req=9f2c01ab session=3b1d7c90 segment=start node=test_planner | qa_agent.graph.instrumentation | Node test_planner finished in 842 ms
```

To see a single run's whole story, run `grep "run=5455144e"` over the backend log. With `LOG_FORMAT=json`, each line is a JSON object with `run_id`, `request_id`, `client_session`, `project`, `segment` and `node` fields, ready for a log shipper.

### Logging

- **Levels:** `LOG_LEVEL` (default `INFO`) and `LOG_FORMAT` (`text` | `json`) are set in `.env`.
- **What gets logged:**
  - Node durations and failures.
  - Segment start and end, with status and outcome.
  - Each API/UI test result.
  - Each knowledge search: hit count and top score.
  - LLM call duration and token usage, but never the prompt or response text.
  - Database sync.
  - HTTP requests: GET polling at `DEBUG`, changes at `INFO`, 4xx/5xx at `WARNING`/`ERROR`.
- Uvicorn's own access log is replaced by the request middleware, which knows the ids. `httpx`, `openai` and `langsmith` (library loggers that can echo request details) are held at `WARNING`.

### Secrets are never logged or traced

- `OPENAI_API_KEY`, `LANGSMITH_API_KEY` and `API_TEST_AUTH_TOKEN` are `SecretStr` settings, so a printed settings object shows `**********`.
- Every formatted log line passes through `redact_text`, and that includes tracebacks and JSON logs. It masks:
  - the exact configured secret values, plus the password in `DATABASE_URL`;
  - common credential shapes: `sk-...`, `lsv2_...`, `Bearer ...`, `user:password@` in URLs, and `api_key=...`/`access_token: ...` pairs.
- Traced tool inputs and outputs go through `redact`. It does the same text masking and also blanks credential-bearing keys (`Authorization`, `Cookie`, `Set-Cookie`, `X-API-Key`, `api_key`, ...).
- The API test tool adds its bearer token only to the outgoing HTTP request. The token never reaches the traced request object, and logged URLs drop the query string.
- `/system/status` reports only whether keys are configured, never their values.
- Deliberately **not** redacted: test data such as a test user's email, a reset token fixture or `new_password` in a request body. These are what a failed test needs to be debugged.

### Verifying locally

- `pytest tests/unit/test_observability.py` captures traces with a mocked LangSmith client and checks:
  - one trace per segment with the run id metadata;
  - node, LLM, retriever and tool runs, and the `node_failure` span;
  - redaction in logs, tracebacks and traces;
  - ids flowing from HTTP headers into background-worker log lines.
- For a live check, set the variables above, run one QA run, then check two things. In LangSmith, the project should show `qa_run.start` (and later `qa_run.execute`) traces. Running `grep` for the run id over the backend log should show every layer.

---

## 18. Running locally on Windows without Docker

Use this when Docker is not available, for example when hardware virtualization is disabled or WSL 2 cannot start. It gives the same stack as Compose: PostgreSQL, FastAPI, Streamlit and the Demo Shop, with health checks. It needs no admin rights.

**One-time setup** (PowerShell, in the repo folder):

```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
.\.venv\Scripts\python -m playwright install chromium
Copy-Item .env.example .env        # then set OPENAI_API_KEY (+ OPENAI_BASE_URL / LLM_MODEL)
.\scripts\setup-postgres.ps1       # portable PostgreSQL 16 into .local\pgsql (~140 MB, no installer)
```

**Start / stop:**

```powershell
.\scripts\start-local.ps1          # Postgres -> migrations -> index -> demo, api, ui (waits for health)
.\scripts\stop-local.ps1           # stops everything (-KeepDatabase leaves Postgres running)
```

`start-local.ps1` does the following:

1. Initialises `.local\pgdata` on the first run. The password is taken from `POSTGRES_PASSWORD` and the server listens on `127.0.0.1` only, with SCRAM authentication.
2. Starts PostgreSQL, creates `qa_agent` and `qa_agent_test`, and runs `alembic upgrade head`.
3. Builds the knowledge base index if it does not exist yet.
4. Starts Demo Shop (`:8765`), FastAPI (`:8000`) and Streamlit (`:8501`) in the background with `DATABASE_URL` pointing at that PostgreSQL, and waits for each health endpoint.

The script is safe to run again: services that are already running are left alone. Logs are written to `logs\` and process ids to `.local\pids.json`. Ports and credentials come from `.env` (`POSTGRES_*`, `API_PORT`, `STREAMLIT_PORT`, `DEMO_PORT`). To run the PostgreSQL tests against it:

```powershell
$env:TEST_DATABASE_URL = "postgresql+psycopg://qa:qa@127.0.0.1:5432/qa_agent_test"; .\.venv\Scripts\python -m pytest
```

`.local\`, `logs\`, `data\*.db` and `.env` are git-ignored.

## 19. Running with Docker Compose

The whole stack runs locally in containers. You need Docker Desktop (Windows/macOS) or Docker Engine with the Compose plugin v2.24 or later.

| Service | Image target | URL on your machine | Purpose |
|---|---|---|---|
| `postgres` | `postgres:16-alpine` | `127.0.0.1:5432` | Runs, test cases, results, bug reports, reviews |
| `api` | `Dockerfile` → `api` | <http://localhost:8000/app/> (chat), `/docs` (API) | FastAPI + LangGraph workflow + Playwright/Chromium |
| `ui` | `Dockerfile` → `ui` | <http://localhost:8501> | Streamlit dashboard |
| `demo` | `Dockerfile` → `demo` | <http://localhost:8765> | Demo Shop: the system under test |

### Start

```bash
cp .env.example .env          # Windows: Copy-Item .env.example .env
# edit .env: set OPENAI_API_KEY (and OPENAI_BASE_URL / LLM_MODEL for an OpenAI-compatible provider)
docker compose up -d --build
docker compose ps             # wait until every service is "healthy"
```

On first start, the `api` container runs `alembic upgrade head` against Postgres and builds the knowledge base index. Then it starts Uvicorn. The start order is enforced by health checks: `postgres` and `demo` must be healthy before `api` starts, and `api` before `ui`.

### Configuration and secrets

- Compose reads `.env` for variable substitution, and the `api` container receives it at runtime (`env_file`). `.env` is listed in `.dockerignore` and `.gitignore`, so it never ends up in an image or in Git. Check with `docker compose run --rm --entrypoint sh api -c 'ls -a /app'`: there is no `.env` there.
- Inside the compose network, compose overrides these values:
  - `DATABASE_URL` points to `postgres:5432`, built from `POSTGRES_USER`, `POSTGRES_PASSWORD` and `POSTGRES_DB`.
  - `API_TEST_BASE_URL` and `UI_TEST_BASE_URL` point to `http://demo:8765`.
  - The dashboard's `STREAMLIT_API_BASE_URL` points to `http://api:8000`.
- Host ports are published on `127.0.0.1` only. Change them with `API_PORT`, `STREAMLIT_PORT`, `DEMO_PORT` and `POSTGRES_PORT`.
- To test another application instead of the demo, set `DOCKER_API_TEST_BASE_URL` and `DOCKER_UI_TEST_BASE_URL`. Use `host.docker.internal` for an app running on your machine.
- Change the default Postgres password (`POSTGRES_PASSWORD`) before sharing the stack. It only takes effect on a fresh `qa-agent-pgdata` volume.

### Data

| Volume | Contents |
|---|---|
| `qa-agent-pgdata` | PostgreSQL data (survives `docker compose down`) |
| `qa-agent-data` | Chroma knowledge base index, UI test screenshots |
| `qa-agent-demo-data` | Demo Shop user accounts |

`docker compose down -v` deletes the volumes. Delete `qa-agent-data` to rebuild the index, or run `docker compose exec api python -m qa_agent.rag.ingestion --rebuild`.

### Everyday commands

```bash
docker compose logs -f api                                   # follow backend logs (run ids in every line)
docker compose exec postgres psql -U qa -d qa_agent -c "select run_id, status from agent_runs"
docker compose up -d --build api                             # rebuild after code changes
docker compose down                                          # stop (data kept)
```

### Tests

The test suite runs on the host, not in the images (tests are excluded from them):

```bash
pytest                                                                    # unit + integration
TEST_DATABASE_URL=postgresql+psycopg://qa:qa@localhost:5432/qa_agent_test pytest   # also against the compose Postgres
```

---

## Open Questions for You

1. **Target system under test** — do you want a small bundled sample app (a toy API + web page included in this repo) to exercise the API/UI executors against, or should the executors point at an external system you'll provide later? (Recommend: bundle a minimal sample Flask/FastAPI + static HTML app under `sample_target_app/` purely as a demo target — keeps the project self-contained and runnable out of the box.)
2. **LLM provider** — OpenAI directly, or a specific OpenAI-compatible endpoint (Azure OpenAI, local vLLM, etc.) you want config defaults tuned for?
3. **Package manager** — `uv`, `poetry`, or plain `pip` + `requirements.txt`? (Recommend `uv` for speed and modern lockfile hygiene, but happy to match what you're used to.)
4. Any preference on Chroma vs FAISS as the *default* (both will be supported behind the interface either way)?

I'll proceed with my recommendations above unless you tell me otherwise — let me know if the architecture looks right, or what you'd like changed, before I start Phase 0.
