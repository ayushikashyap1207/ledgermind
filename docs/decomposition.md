# `needs_both` decomposition

Date: 2026-11-08

## Problem

After the router v2 change, routing was **30/30**, but five `needs_both` cases
still had a document citation and no SQL citation. The SQL generator returned
`NOT_ANSWERABLE` for compound questions such as:

> "How many pending transactions and what approval threshold applies?"

The model treated the policy half as making the entire question unanswerable,
even though the first half is a perfectly valid ledger query. Router-only eval
was 10/10 on route accuracy but 6/10 on route + citations for this reason.

## What changed

- Added `app/router/decompose.py`:
  - LLM structured output splits a compound question into `sql_question` and
    `rag_question`;
  - parses plain or fenced JSON;
  - rejects empty, identical, or SQL-statement outputs;
  - falls back to a conjunction-based heuristic split when the provider fails.
- `app/router/pipeline.py` now uses the decomposed sub-questions **only** for
  `needs_both`:
  - `answer_from_sql(sql_question)`
  - `retrieve(rag_question, top_k=...)`
  - synthesis still receives the original question.
- The SQL safety layer, read-only execution, and `needs_sql` / `needs_rag`
  paths are unchanged.
- `POST /query` now exposes `sql_question`, `rag_question`, and
  `decomposition_method`.
- Evals record the decomposition metadata per case.
- Added `tests/test_router_decompose.py` (7 tests). Total suite at that point:
  **25 tests** (34 after the hybrid-retrieval tests were added).
- The faithfulness judge no longer crashes the eval on a provider error:
  `answer_faithfulness` now reports `judged`, `errors`, and a `rate` over
  successfully judged cases only.

## Validation so far

| Check | Result |
|---|---|
| `pytest` | **25/25 passed** |
| Router eval with decomposition | **10/10 passed** (route accuracy 10/10) |
| Router eval before decomposition | 6/10 (route accuracy 10/10) |
| Smoke test: 5 `needs_both` SQL sub-questions | **5/5 returned `ok`** with a result row |

Example decompositions from the smoke test:

| Original | SQL sub-question | RAG sub-question |
|---|---|---|
| How many pending transactions and what approval threshold applies? | Count of pending transactions in the transactions table. | What is the approval threshold for transactions? |
| Total overdue invoices and MSA payment terms? | What is the total amount of overdue invoices in USD as of 2026-11-07? | What are the payment terms specified in the company's Master Service Agreements? |
| Vendor count in Germany and retention expectation? | Count of vendors in Germany | Retention expectation for vendors in Germany |
| Top vendor by invoiced amount plus concentration cap? | Which vendor has the highest total invoiced amount for the period 2025-01-01 to 2025-12-31? | What is the concentration cap for vendor spend...? |
| Void transaction total and unresolved audit exception cap? | SELECT SUM(amount) AS total_voided FROM transactions WHERE status='void'; | What is the maximum allowed number of unresolved audit exceptions...? |

The last two rows exposed two prompt issues, both now addressed:

- the model invented a date range that was not in the original question;
- the model returned a SQL statement instead of a natural-language question.

The prompt now explicitly forbids invented dates/periods and SQL statements,
and the parser rejects SQL-looking `sql_question` values.

## Full 30-case eval: measured with a quota workaround

The configured `gpt-oss-20b` model hit Groq's 200k-token daily limit during
repeated runs. A complete post-decomposition + hybrid-retrieval run was
therefore executed with `LLM_MODEL=openai/gpt-oss-120b`:

| Metric | Result |
|---|---:|
| Routing accuracy | **30/30 (100%)** |
| Citation correctness | **30/30 (100%)** |
| Answer faithfulness | **27/30 (90.0%)** |
| SQL execution correctness | 11/15 (73.3%) |
| Retrieval recall@3 | 15/15 (100%) |

All five `needs_both` cases used LLM decomposition and produced both SQL and
document citations. The remaining `e28` retrieval miss was fixed separately by
hybrid BM25 + dense retrieval; see `docs/hybrid_retrieval.md`.

The 120b run is not directly comparable to the 20b baseline, but it validates
the end-to-end behavior. A 20b re-run is still pending quota.

## Follow-ups

1. Re-run the full eval on `gpt-oss-20b` after its daily quota resets.
2. Add provider retry/backoff and a per-run token budget so evals do not exhaust
   the daily quota.
3. Add a decomposition-quality check: reject sub-questions that introduce
   numbers, dates, or entities absent from the original question.
4. Fix the remaining SQL failures (overdue definition, join qualification,
   projection strictness).
