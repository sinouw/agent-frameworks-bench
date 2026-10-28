# Results

Answer model `openai/gpt-oss-120b`, judge `qwen/qwen3.8-27b`.

## End to end (14 questions, each answered by all three)

Latency, LLM calls, tokens, prompt size and cost cover the 12 questions that all three answered without an error.

| Framework | Correct | Faithful | Citation precision | Tool selection | p50 / p95 latency | p50 non-LLM time | LLM calls | Tokens in / out | Prompt chars sent | Cost / question | Errors | 429s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline | 93% | 100% | 94% | 93% | 1,180 / 3,067 ms | 95 ms | 2.2 | 2,379 / 257 | 11,484 | $0.00051 | 0 | 3 |
| langgraph | 79% | 100% | 94% | 93% | 1,533 / 1,818 ms | 130 ms | 2.2 | 2,079 / 244 | 9,959 | $0.00046 | 2 | 3 |
| llamaindex | 100% | 100% | 94% | 93% | 1,613 / 2,682 ms | 192 ms | 2.3 | 2,330 / 253 | 11,394 | $0.00050 | 0 | 2 |

### By category

| Category | baseline | langgraph | llamaindex |
|---|---|---|---|
| calculation (2) | 50% correct, 50% tool, 0 errors | 50% correct, 50% tool, 0 errors | 100% correct, 50% tool, 0 errors |
| lookup (6) | 100% correct, 100% tool, 0 errors | 100% correct, 100% tool, 0 errors | 100% correct, 100% tool, 0 errors |
| metadata_sql (2) | 100% correct, 100% tool, 0 errors | 100% correct, 100% tool, 0 errors | 100% correct, 100% tool, 0 errors |
| multi_doc (2) | 100% correct, 100% tool, 0 errors | 100% correct, 100% tool, 0 errors | 100% correct, 100% tool, 0 errors |
| unanswerable (2) | 100% correct, 100% tool, 0 errors | 0% correct, 100% tool, 2 errors | 100% correct, 100% tool, 0 errors |

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
| shared | 536 |

## Indexing (25 documents, 1251 chunks incl. metadata cards)

| Implementation | Seconds |
|---|---|
| embed corpus once (dense + BM25, shared) | 285.57 |
| baseline: write index | 3.37 |
| langchain: write index (x2: default and IDF collections) | 7.48 |
| llamaindex: write index | 3.64 |
