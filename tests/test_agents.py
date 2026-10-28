"""Each framework's real agent code, run against a scripted OpenAI-compatible backend through
the metered transport. No network, no API key."""

import pytest
from pydantic import SecretStr

from afbench.config import Settings
from afbench.stack import MATCHED, build_bench
from afbench.transport import Meter
from tests.conftest import ScriptedBackend, StubSearcher, completion

URL = "https://llm.test/v1"


def make_agent(framework, backend: ScriptedBackend, meter: Meter, searcher: StubSearcher, db, max_steps: int = 5):
    """Agents built by the production wiring, talking to the scripted backend."""
    settings = Settings(
        groq_api_key=SecretStr("test"), llm_base_url=URL, answer_model="test-model", sqlite_path=db, max_steps=max_steps
    )
    bench = build_bench(settings, dict.fromkeys(MATCHED.values(), searcher), (framework,), backend.transport())
    bench.meter.calls = meter.calls  # share the call log with the caller
    return bench.agents[framework]


FRAMEWORKS = ["baseline", "langgraph", "llamaindex"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_search_then_cited_answer(framework, metadata_db, stub_searcher):
    backend = ScriptedBackend(
        [
            completion(tool_calls=[("search_docs", {"query": "max line length"})]),
            completion("Limit lines to 79 characters [S1]."),
        ]
    )
    meter = Meter()
    result = await make_agent(framework, backend, meter, stub_searcher, metadata_db).answer("Max line length?")

    assert result.error is None
    assert result.answer == "Limit lines to 79 characters [S1]."
    assert [t.name for t in result.tools] == ["search_docs"]
    assert stub_searcher.queries == ["max line length"]
    assert [h.doc_id for h in result.cited] == ["pep-0008"]
    assert len(meter.calls) == 2
    # Every framework offers the same three tools with the same names.
    first = backend.requests[0]
    assert sorted(t["function"]["name"] for t in first["tools"]) == ["calculator", "search_docs", "sql_query"]
    # The tool result goes back to the model with its [S1] label.
    tool_messages = [m for m in backend.requests[1]["messages"] if m["role"] == "tool"]
    assert "[S1] PEP 8" in tool_messages[0]["content"]


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_sql_and_calculator_tools(framework, metadata_db, stub_searcher):
    backend = ScriptedBackend(
        [
            completion(tool_calls=[("sql_query", {"sql": "SELECT count(*) AS n FROM documents"})]),
            completion(tool_calls=[("calculator", {"expression": "2 * 21"})]),
            completion("There are 2 documents; twice 21 is 42."),
        ]
    )
    result = await make_agent(framework, backend, Meter(), stub_searcher, metadata_db).answer("q")
    assert [(t.name, t.output) for t in result.tools] == [("sql_query", "n\n2"), ("calculator", "2 * 21 = 42")]
    assert result.answer.endswith("42.")


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_provider_error_becomes_result_error(framework, metadata_db, stub_searcher):
    backend = ScriptedBackend([])
    backend.handler = lambda request: __import__("httpx").Response(500, json={"error": {"message": "boom"}})
    result = await make_agent(framework, backend, Meter(), stub_searcher, metadata_db).answer("q")
    assert result.error is not None and result.answer == ""


async def test_step_budget_differs_by_design(metadata_db, stub_searcher):
    """A model that never stops calling tools: the baseline and LlamaIndex force a final answer
    on the last step; the prebuilt LangGraph agent stops with a recursion error."""
    looping = [completion(tool_calls=[("calculator", {"expression": "1 + 1"})])] * 10
    outcomes = {}
    for framework in FRAMEWORKS:
        backend = ScriptedBackend([*looping[:2], completion("Final answer."), *looping])
        if framework == "langgraph":
            backend = ScriptedBackend(list(looping))
        result = await make_agent(framework, backend, Meter(), stub_searcher, metadata_db, max_steps=3).answer("q")
        outcomes[framework] = (result.answer, result.error)
    assert outcomes["baseline"] == ("Final answer.", None)
    assert outcomes["llamaindex"][1] is None
    assert outcomes["langgraph"] == ("", "no answer within the step budget")


@pytest.mark.parametrize("framework", FRAMEWORKS)
async def test_same_model_and_temperature_on_the_wire(framework, metadata_db, stub_searcher):
    backend = ScriptedBackend([completion("ok")])
    await make_agent(framework, backend, Meter(), stub_searcher, metadata_db).answer("q")
    request = backend.requests[0]
    assert (request["model"], request.get("temperature")) == ("test-model", 0.0)
    assert request["messages"][0]["role"] == "system"
