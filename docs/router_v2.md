# Router v2 — LLM classifier

Date: 2026-11-07
Baseline compared against: `docs/baseline.md` (commit `28d2761` + fallback removal)

> **Follow-up (2026-11-08):** `needs_both` decomposition is implemented.
> See `docs/decomposition.md`. The full post-decomposition eval is pending a
> Groq daily-token reset.

## What changed

- `app/router/classifier.py` now uses an LLM structured-output classifier:
  - returns `{"route", "confidence", "reason"}` for
    `needs_sql | needs_rag | needs_both`;
  - parses plain JSON and markdown-fenced JSON;
  - falls back to the deterministic keyword scorer when the provider fails
    or returns an unparseable/invalid response;
  - unknown keyword-fallback questions now default to `needs_both` instead of
    `needs_sql`.
- `QueryAnswer` and `POST /query` now expose `route_confidence` and
  `route_method` (`llm` or `keyword`).
- Router and end-to-end evals record the confidence and method per case.
- Added `tests/test_router_classifier.py` (6 tests: valid JSON, fenced JSON,
  malformed JSON, invalid route, provider error, unknown-question fallback).
- No changes to SQL generation, SQL safety, or execution.

## Results — same 30-case end-to-end suite, same Groq model

| Metric | Baseline | Router v2 | Delta |
|---|---:|---:|---:|
| Routing accuracy | 18/30 (60.0%) | **30/30 (100.0%)** | **+12** |
| Citation correctness | 14/30 (46.7%) | **24/30 (80.0%)** | **+10** |
| Answer faithfulness | 19/30 (63.3%) | **26/30 (86.7%)** | **+7** |
| SQL execution correctness | 12/15 (80.0%) | 11/15 (73.3%)* | −1 (variance) |
| Latency p50 | 4,567 ms | 5,680 ms | +1,113 ms |
| Latency mean | 3,452 ms | 6,024 ms | +2,572 ms |
| Latency max | 7,258 ms | 10,047 ms | +2,789 ms |

\* SQL generation is unchanged. A standalone rerun produced 12/15, the full
eval produced 11/15, and a third rerun produced 10/15 with one Groq `429`
rate-limit failure. The variance comes from model nondeterminism, strict
row/column comparison, and rate limiting — not from the router change.

Router-only eval (`eval/router_eval.py`): **10/10 route accuracy**. The full
pass count is 6/10 because four `needs_both` cases still lack a SQL citation.

Raw artifacts:

- `docs/baseline/full_eval_router_v2.json`
- `docs/baseline/router_eval_llm.json`
- `docs/baseline/sql_eval_router_v2.json`

## Remaining failures

### 1. `needs_both` has no SQL citation (5 cases: e21–e25)

The router is now correct, but the SQL generator returns `NOT_ANSWERABLE` for
compound questions such as:

> "How many pending transactions and what approval threshold applies?"

The model treats the policy half as making the whole question unanswerable
instead of extracting the SQL-answerable half. Fix: decompose `needs_both`
questions into a SQL sub-question and a document sub-question, run each
pipeline on its own sub-question, then synthesize.

This is the next highest-leverage change: it accounts for 5 of the 6 remaining
citation failures and should lift faithfulness further.

### 2. `e28` retrieval miss

> "What is the reimbursement timeline for approved expenses?"

Top-3 sources were `procurement_policy.txt`, `vendor_msa_excerpt.txt`, and
`earnings_call_q2_2026.txt`; `expense_policy.txt` was not retrieved. This is
the hash-embedding limitation showing up. Fix with real embeddings plus a
reranker.

### 3. SQL reliability

- `q02` defines "overdue" as `due_date < now` instead of `status = 'overdue'`.
- `q06` and `q11` fail the strict column-set comparison (semantically close
  answers, different projection).
- `q15` generates an unqualified `vendor_id` join and the wrong overdue
  definition.
- Groq rate limits appeared during back-to-back eval runs. Add retry/backoff
  and a timeout before relying on these numbers operationally.

## Trade-off

The LLM classifier adds one provider call per question, which is visible in
latency (p50 +1.1 s). For production, this can be reduced later by skipping the
LLM when the keyword scorer is highly confident, or by using a smaller/faster
routing model. Correctness first; optimize after the decomposition work.

## Verification

- `pytest`: **18/18 passed**
- App health: `app`, `database`, `chroma` all `ok`
- `POST /query` response is backward compatible; two new fields were added.
