"""Command line: `uv run afbench <command>` (or `python -m afbench`).

index       build the three indexes (fastembed + Qdrant, no API key)
retrieval   retrieval metrics for every variant over all golden questions (no API key)
budget      estimate the tokens a resumed e2e + judge pass will spend, per model (no API calls)
probe       one tiny request per model: is its daily window free? (needs GROQ_API_KEY)
e2e         end-to-end agent eval on Groq, resumable (needs GROQ_API_KEY)
judge       grade the recorded answers, one at a time, resumable (needs GROQ_API_KEY)
report      write results/summary.{json,md} from the e2e rows, retrieval and LOC
loc         count lines of code per implementation
ask         ask one framework one question
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from afbench import evals, quota, report
from afbench.config import REPO_ROOT, Settings, get_settings
from afbench.loc import count_files
from afbench.stack import DAILY_WAIT_S, FRAMEWORKS, build_bench, build_indexes, open_searchers, write_json

IMPLEMENTATION_FILES = {
    "baseline": ["frameworks/baseline.py"],
    "langgraph": ["frameworks/langgraph_impl.py"],
    "llamaindex": ["frameworks/llamaindex_impl.py"],
    "shared": ["common.py", "corpus.py", "embeddings.py"],
}


def cmd_loc() -> dict[str, dict[str, int]]:
    src = REPO_ROOT / "src" / "afbench"
    return {name: count_files([src / f for f in files]) for name, files in IMPLEMENTATION_FILES.items()}


def _subset(settings: Settings, args: argparse.Namespace) -> list[evals.GoldenItem]:
    items = evals.load_golden(settings.golden_path)
    if getattr(args, "ids", None):
        wanted = set(args.ids.split(","))
        return [i for i in items if i.id in wanted]
    return items if getattr(args, "all", False) else evals.select_subset(items)


def _quota_ok(settings: Settings, model: str, args: argparse.Namespace) -> bool:
    """Probe the model before a pass; wait for its daily window if asked to, else stop."""
    max_wait = args.wait_for_quota * 3600 if args.wait_for_quota else DAILY_WAIT_S
    result = asyncio.run(quota.wait_for_quota(settings.llm_base_url, settings.api_key(), model, max_wait))
    if result.status != 200:
        print(f"not starting: {quota.describe(result)}. Rerun later, or pass --wait-for-quota HOURS.")
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="afbench")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    sub.add_parser("retrieval")
    e2e = sub.add_parser("e2e")
    e2e.add_argument("--run", default="main", help="run name; rows go to results/e2e/<run>.jsonl")
    e2e.add_argument("--all", action="store_true", help="all 65 questions instead of the fixed subset")
    e2e.add_argument("--ids", help="comma-separated question ids")
    e2e.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    judge = sub.add_parser("judge")
    judge.add_argument("--run", default="main")
    for p in (e2e, judge):
        p.add_argument("--wait-for-quota", type=float, metavar="HOURS", help="wait up to HOURS for the daily window")
    budget = sub.add_parser("budget")
    budget.add_argument("--run", default="main")
    budget.add_argument("--frameworks", default=",".join(FRAMEWORKS))
    pr = sub.add_parser("probe")
    pr.add_argument("--model", action="append", help="default: the answer and judge models")
    rep = sub.add_parser("report")
    rep.add_argument("--run", default="main")
    sub.add_parser("loc")
    ask = sub.add_parser("ask")
    ask.add_argument("framework", choices=FRAMEWORKS)
    ask.add_argument("question")
    args = parser.parse_args(argv)

    settings = get_settings()
    results = settings.results_dir
    out = results / "e2e" / f"{getattr(args, 'run', 'main')}.jsonl"
    if args.command == "index":
        write_json(results / "index.json", build_indexes(settings))
    elif args.command == "retrieval":
        items = evals.load_golden(settings.golden_path)
        data = evals.retrieval_eval(open_searchers(settings), items, settings.top_k)
        write_json(results / "retrieval.json", data)
    elif args.command == "budget":
        frameworks = tuple(f for f in args.frameworks.split(",") if f)
        plan = quota.estimate(
            _subset(settings, args), evals.load_rows(out), frameworks, settings.answer_model, settings.judge_model
        )
        write_json(results / "budget.json", plan)
        a, j = plan["answer"], plan["judge"]
        print(f"answers: {len(a['pending_runs'])} runs on {a['model']}, ~{a['estimated_tokens']:,} tokens")
        print(f"judge:   {j['calls']} calls on {j['model']}, ~{j['estimated_tokens']:,} tokens")
        print(f"daily cap {quota.DAILY_TOKENS:,} per model; fits one window: {plan['fits_one_daily_window']}")
    elif args.command == "probe":
        for model in args.model or [settings.answer_model, settings.judge_model]:
            print(
                quota.describe(asyncio.run(quota.wait_for_quota(settings.llm_base_url, settings.api_key(), model, 0)))
            )
    elif args.command == "e2e":
        items = _subset(settings, args)
        frameworks = tuple(f for f in args.frameworks.split(",") if f)
        pending = [i for i in items if any(not evals.final(evals.load_rows(out).get((f, i.id))) for f in frameworks)]
        if not pending:
            print("nothing to answer")
            return 0
        if not _quota_ok(settings, settings.answer_model, args):
            return 3
        bench = build_bench(settings, open_searchers(settings), frameworks)
        print(f"{len(items)} questions x {len(bench.agents)} frameworks, answer model {settings.answer_model}")
        completed = asyncio.run(evals.answer_all(bench.agents, bench.meter, items, settings.answer_model, out))
        return 0 if completed else 3
    elif args.command == "judge":
        items = evals.load_golden(settings.golden_path)
        if not _quota_ok(settings, settings.judge_model, args):
            return 3
        bench = build_bench(settings, {}, ())
        completed = asyncio.run(evals.judge_all(bench.judge, bench.meter, items, settings.judge_model, out))
        return 0 if completed else 3
    elif args.command == "report":
        write_json(results / "loc.json", cmd_loc())
        summary = report.build(settings, args.run)
        write_json(results / "summary.json", summary)
        (results / "summary.md").write_text(report.markdown(summary))
        print(f"wrote {results / 'summary.md'}")
    elif args.command == "loc":
        print(json.dumps(cmd_loc(), indent=2))
    elif args.command == "ask":
        bench = build_bench(settings, open_searchers(settings), (args.framework,))
        result = asyncio.run(bench.agents[args.framework].answer(args.question))
        print(result.answer or f"error: {result.error}")
        for tool in result.tools:
            print(f"  tool {tool.name} {json.dumps(tool.arguments)}", file=sys.stderr)
        for call in bench.meter.calls:
            print(f"  llm {call.input_tokens}/{call.output_tokens} tokens, {call.tools} tools offered", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
