"""Command line: `uv run afbench <command>` (or `python -m afbench`).

index       build the three indexes (fastembed + Qdrant, no API key)
retrieval   retrieval metrics for every variant over all golden questions (no API key)
e2e         end-to-end agent eval on Groq, resumable (needs GROQ_API_KEY)
report      write results/summary.{json,md} from the e2e rows, retrieval and LOC
loc         count lines of code per implementation
ask         ask one framework one question
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from afbench import evals, report
from afbench.config import REPO_ROOT, get_settings
from afbench.loc import count_files
from afbench.stack import FRAMEWORKS, build_bench, build_indexes, open_searchers, write_json

IMPLEMENTATION_FILES = {
    "baseline": ["frameworks/baseline.py"],
    "langgraph": ["frameworks/langgraph_impl.py"],
    "llamaindex": ["frameworks/llamaindex_impl.py"],
    "shared": ["common.py", "corpus.py", "embeddings.py"],
}


def cmd_loc() -> dict[str, dict[str, int]]:
    src = REPO_ROOT / "src" / "afbench"
    return {name: count_files([src / f for f in files]) for name, files in IMPLEMENTATION_FILES.items()}


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
    elif args.command == "e2e":
        items = evals.load_golden(settings.golden_path)
        if args.ids:
            wanted = set(args.ids.split(","))
            items = [i for i in items if i.id in wanted]
        elif not args.all:
            items = evals.select_subset(items)
        frameworks = tuple(f for f in args.frameworks.split(",") if f)
        bench = build_bench(settings, open_searchers(settings), frameworks)
        print(f"{len(items)} questions x {len(bench.agents)} frameworks, answer model {settings.answer_model}")
        completed = asyncio.run(evals.answer_all(bench.agents, bench.meter, items, settings.answer_model, out))
        return 0 if completed else 3
    elif args.command == "judge":
        items = evals.load_golden(settings.golden_path)
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
