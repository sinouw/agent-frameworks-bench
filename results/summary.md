# Results

Answer model `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`.

## End to end (10 questions, each answered by all three)

Latency, LLM calls, tokens, prompt size and cost cover the 9 questions that all three answered without an error.

| Framework | Correct | Faithful | Citation precision | Tool selection | p50 / p95 latency | p50 non-LLM time | LLM calls | Tokens in / out | Prompt chars sent | Cost / question | Errors | 429s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | pending (0/10 judged) | pending (0/10 judged) | 92% | 90% | 1,215 / 3,094 ms | 138 ms | 2.3 | 2,506 / 280 | 12,063 | $0.00054 | 0 | 3 |
| langgraph | pending (0/9 judged) | pending (0/9 judged) | 92% | 90% | 1,533 / 2,816 ms | 139 ms | 2.2 | 2,101 / 236 | 10,050 | $0.00046 | 1 | 3 |
| llamaindex | pending (0/10 judged) | pending (0/10 judged) | 92% | 90% | 1,613 / 3,454 ms | 192 ms | 2.3 | 2,276 / 255 | 11,169 | $0.00049 | 0 | 2 |

### By category

| Category | baseline | langgraph | llamaindex |
|---|---|---|---|
| calculation (2) | 50% tool, 0 errors | 50% tool, 0 errors | 50% tool, 0 errors |
| lookup (3) | 100% tool, 0 errors | 100% tool, 0 errors | 100% tool, 0 errors |
| metadata_sql (2) | 100% tool, 0 errors | 100% tool, 0 errors | 100% tool, 0 errors |
| multi_doc (2) | 100% tool, 0 errors | 100% tool, 0 errors | 100% tool, 0 errors |
| unanswerable (1) | 100% tool, 0 errors | 0% correct, 100% tool, 1 errors | 100% tool, 0 errors |

## Retrieval (all golden questions, no LLM)

| Variant | Doc Hit@1 | Doc Hit@5 | Doc MRR | Answer-chunk Hit@5 | Answer-chunk MRR | Chunk MRR core / hard | p50 latency |
|---|---|---|---|---|---|---|---|
| baseline | 96% | 100% | 0.98 | 89% | 0.72 | 0.88 / 0.56 | 82 ms |
| langchain: defaults | 96% | 100% | 0.98 | 89% | 0.68 | 0.83 / 0.52 | 82 ms |
| langchain: IDF | 96% | 100% | 0.98 | 89% | 0.72 | 0.86 / 0.57 | 78 ms |
| langchain: IDF + 20 candidates | 96% | 100% | 0.98 | 89% | 0.72 | 0.88 / 0.56 | 84 ms |
| llamaindex: defaults | 96% | 100% | 0.98 | 89% | 0.72 | 0.88 / 0.56 | 82 ms |
| llamaindex: 20 candidates | 96% | 100% | 0.97 | 91% | 0.74 | 0.90 / 0.57 | 95 ms |
| llamaindex: 20 candidates + BM25 query encoding | 96% | 100% | 0.97 | 91% | 0.74 | 0.90 / 0.57 | 74 ms |
| llamaindex: 20 candidates + BM25 query encoding + RRF | 96% | 100% | 0.98 | 89% | 0.72 | 0.88 / 0.56 | 85 ms |

## Lines of code (logical lines: no blanks, comments or docstrings)

| Part | Lines |
|---|---|
| baseline | 140 |
| langgraph | 107 |
| llamaindex | 140 |
| shared | 531 |

## Indexing (25 documents, 1251 chunks incl. metadata cards)

| Implementation | Seconds |
|---|---|
| embed corpus once (dense + BM25, shared) | 285.57 |
| baseline: write index | 3.37 |
| langchain: write index (x2: default and IDF collections) | 7.48 |
| llamaindex: write index | 3.64 |
