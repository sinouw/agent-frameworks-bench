import pytest

from afbench.common import (
    CalculatorError,
    Hit,
    SourceRegistry,
    SqlTool,
    ToolBox,
    calculator,
    extract_citation_labels,
    format_hits,
    normalize_citations,
    safe_eval,
)
from tests.conftest import PEP8_HIT, StubSearcher


def test_calculator_arithmetic_and_functions():
    assert safe_eval("(2023 - 2001) * 12") == 264
    assert safe_eval("max(3, 7) + abs(-2)") == 9
    assert calculator("79 - 72") == "79 - 72 = 7"
    assert calculator("10 / 4") == "10 / 4 = 2.5"


@pytest.mark.parametrize("expr", ["__import__('os')", "a + 1", "2 ** 1000", "1 / 0", "max(x=1)", "'a' * 3"])
def test_calculator_rejects_unsafe_or_invalid(expr):
    with pytest.raises(CalculatorError):
        safe_eval(expr)
    assert calculator(expr).startswith("Error:")


def test_sql_tool_reads_and_refuses_writes(metadata_db):
    sql = SqlTool(metadata_db)
    assert "CREATE TABLE documents" in sql.schema()
    out = sql.run("SELECT pep_number, status FROM documents ORDER BY pep_number")
    assert out.splitlines() == ["pep_number | status", "8 | Active", "572 | Final"]
    assert sql.run("DELETE FROM documents").startswith("Error")
    assert sql.run("SELECT 1; DROP TABLE documents").startswith("Error")
    # A write hidden in a CTE still hits the read-only connection and authorizer.
    assert "error" in sql.run("WITH x AS (SELECT 1) INSERT INTO documents VALUES (1)").lower()
    assert sql.run("SELECT nope FROM documents").startswith("SQL error")


def test_registry_labels_are_stable_across_searches():
    registry = SourceRegistry()
    other = Hit("pep-0020#0", "pep-0020", "PEP 20", "", "Beautiful is better than ugly.", 0.5)
    first = format_hits([PEP8_HIT, other], registry)
    second = format_hits([other], registry)
    assert first.startswith("[S1] PEP 8: Style Guide for Python Code › Maximum Line Length")
    assert second.startswith("[S2] PEP 20")
    assert format_hits([], registry) == "No matching passages found."


def test_citation_parsing():
    answer = normalize_citations("Use 79 characters 【S1】, see also [S2, S3] and [S1].")
    assert extract_citation_labels(answer) == ["S1", "S2", "S3"]


def test_toolbox_logs_calls_and_resolves_citations(metadata_db):
    box = ToolBox(StubSearcher([PEP8_HIT]), SqlTool(metadata_db))
    assert "[S1]" in box.search_docs("line length", 3)
    assert box.search_docs("  ").startswith("Error")
    box.calculator("1 + 1")
    result = box.result("79 characters [S1] [S9]")
    assert [t.name for t in result.tools] == ["search_docs", "search_docs", "calculator"]
    assert [h.doc_id for h in result.cited] == ["pep-0008"]  # [S9] doesn't resolve


def test_errors_are_redacted(metadata_db):
    box = ToolBox(StubSearcher([]), SqlTool(metadata_db))
    result = box.result("", error="429 for organization `org_0123456789abcdef` key gsk_abcdef123456")
    assert result.error is not None
    assert "org_0123" not in result.error and "gsk_" not in result.error
