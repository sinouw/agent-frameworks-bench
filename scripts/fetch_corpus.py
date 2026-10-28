"""Download the demo corpus: a fixed set of Python Enhancement Proposals.

PEPs are public domain / CC0-1.0, so the files are committed to the repo and
this script only needs to run when the PEP list changes. The source commit is
pinned so the corpus (and therefore the eval numbers) stay reproducible.
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

PEPS_COMMIT = "06b1b26d959e11bba44b70e5b6f5964bde50eb48"
PEP_NUMBERS = [
    8, 20, 257, 343, 380, 484, 492, 498, 518, 526, 557, 572, 584,
    604, 618, 634, 657, 668, 695, 701, 703, 709, 723, 3107, 3333,
]

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for number in PEP_NUMBERS:
        name = f"pep-{number:04d}.rst"
        url = f"https://raw.githubusercontent.com/python/peps/{PEPS_COMMIT}/peps/{name}"
        with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310 - fixed https URL
            (OUT_DIR / name).write_bytes(resp.read())
        print(f"fetched {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
