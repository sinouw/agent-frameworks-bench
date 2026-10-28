import json

import httpx
from openai import AsyncOpenAI

from afbench import evals
from afbench.common import AgentResult, Hit, ToolUse
from afbench.config import REPO_ROOT
from afbench.loc import logical_lines
from afbench.transport import Meter, metered_client
from tests.conftest import PEP8_HIT, completion


def test_golden_set_and_subset_rule():
    items = evals.load_golden(REPO_ROOT / "evals" / "golden.jsonl")
    assert len(items) == 65
    subset = evals.select_subset(items)
    counts: dict[tuple[str, str], int] = {}
    for item in subset:
        key = (item.category, item.difficulty if item.category == "lookup" else "")
        counts[key] = counts.get(key, 0) + 1
    assert counts == {
        ("lookup", "core"): 3,
        ("lookup", "hard"): 3,
        ("multi_doc", ""): 2,
        ("metadata_sql", ""): 2,
        ("calculation", ""): 2,
        ("unanswerable", ""): 2,
    }
    assert {i.tool for i in subset if i.category == "metadata_sql"} == {"sql_query"}


def test_retrieval_metrics():
    items = [
        evals.GoldenItem("a", "lookup", "q", "r", ["pep-0008"], "79 characters"),
        evals.GoldenItem("b", "lookup", "q", "r", ["pep-0020"], None),
    ]

    class Fixed:
        name = "fixed"

        def search(self, query: str, k: int) -> list[Hit]:
            other = Hit("pep-0020#1", "pep-0020", "PEP 20", "", "zen", 0.1)
            return [other, PEP8_HIT]

    s = evals.retrieval_summary(evals.retrieval_rows(Fixed(), items))
    assert s["hit_at_1"] == 0.5 and s["hit_at_5"] == 1.0
    assert s["mrr"] == (0.5 + 1.0) / 2
    assert s["evidence_hit_at_5"] == 1.0 and s["evidence_mrr"] == 0.5


def test_parse_judgement():
    assert evals.parse_judgement('<think>hmm {"x":1}</think>{"correct": true, "faithful": false, "reason": "r"}') == {
        "correct": True,
        "faithful": False,
        "reason": "r",
    }
    assert evals.parse_judgement("no json")["correct"] is None


class FakeAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.calls = 0

    async def answer(self, question: str) -> AgentResult:
        self.calls += 1
        return AgentResult("79 characters [S1].", [ToolUse("search_docs", {"query": question}, "[S1] ...")], [PEP8_HIT])


async def test_answer_and_judge_passes_cover_all_pairs_and_resume(tmp_path):
    items = [evals.GoldenItem("q01", "lookup", "Max line length?", "79", ["pep-0008"])]
    agents = {name: FakeAgent(name) for name in ("baseline", "langgraph", "llamaindex")}
    meter = Meter()
    judge_bodies = []

    def judge_handler(request: httpx.Request) -> httpx.Response:
        judge_bodies.append(json.loads(request.content))
        return httpx.Response(200, json=completion('{"correct": true, "faithful": true, "reason": "ok"}'))

    judge = AsyncOpenAI(
        api_key="t",
        base_url="https://j.test/v1",
        max_retries=0,
        http_client=metered_client(meter, httpx.MockTransport(judge_handler)),
    )
    out = tmp_path / "run.jsonl"
    assert await evals.answer_all(agents, meter, items, "answer-model", out)
    assert all(r["correct"] is None for r in evals.load_rows(out).values())
    assert await evals.judge_all(judge, meter, items, "judge-model", out)
    rows = evals.load_rows(out)
    assert len(rows) == 3 and all(r["correct"] and r["tool_ok"] for r in rows.values())
    assert rows[("langgraph", "q01")]["citation_precision"] == 1.0
    assert "[S1] ..." in judge_bodies[0]["messages"][0]["content"]

    # Resuming does not re-run or re-judge finished pairs.
    assert await evals.answer_all(agents, meter, items, "answer-model", out)
    assert await evals.judge_all(judge, meter, items, "judge-model", out)
    assert all(a.calls == 1 for a in agents.values())
    assert len(judge_bodies) == 3

    summary = evals.summarize_run(rows, list(agents))
    assert summary["paired_questions"] == ["q01"]
    assert summary["frameworks"]["baseline"]["correctness"] == 1.0


def test_logical_lines_skip_docstrings_and_comments():
    source = '"""Module doc."""\n\n# comment\ndef f(x):\n    """Doc."""\n    return (x +\n            1)  # trailing\n'
    assert logical_lines(source) == 3
