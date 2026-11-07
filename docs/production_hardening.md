# Production hardening

Date: 2026-11-08

This pass closes the most concrete P0 gaps from the production review. It is a
minimal gate, not a complete compliance program.

## SQL safety

- **Timeout is now enforced** (`app/sql/executor.py`):
  - SQLite uses `sqlite3.Connection.set_progress_handler` with a monotonic
    deadline;
  - Postgres uses `SET statement_timeout`;
  - timeouts raise `QueryTimeout`.
- **Result row cap is enforced** at fetch time: the executor fetches
  `sql_row_limit + 1` rows and raises `QueryResultTooLarge` if the cap is
  exceeded.
- **`SELECT *` is rejected** by the validator. `COUNT(*)` remains allowed.
  Tests cover bare `*` and `t.*`.
- **Function allowlist covers all `exp.Func` nodes**, not just `Anonymous`.
  Typed functions such as `json_extract` no longer bypass the allowlist.
- The read-only connection remains layer 2 defense in depth.

Tests: `tests/test_executor_timeout.py`, updated `tests/test_sql_safety.py`
and `tests/test_executor_readonly.py`.

## API security and audit

- Optional bearer-token auth (`API_KEY`):
  - protects `/query`, `/ingest`, `/ingest/sql`, and `/eval/run`;
  - `/health` and `/docs` stay public for load balancers and local development;
  - timing-safe comparison via `secrets.compare_digest`.
- Audit middleware (`app/core/audit.py`):
  - logs request ID, method, path, status, duration, client, and whether an
    authorization header was present;
  - never logs question text, document content, SQL, or API keys;
  - returns `x-request-id` to the caller.
- Production still needs a write-once audit sink (database/SIEM), SSO/RBAC,
  tenant isolation, TLS, and a secrets manager. Those are intentionally not
  faked with a superficial role header.

Tests: `tests/test_auth.py`.

## Provider resilience

- `RetryingLLMClient` wraps every provider with bounded exponential backoff
  (`LLM_MAX_RETRIES`, `LLM_RETRY_BACKOFF_SECONDS`).
- Retries only transient failures: 408/409/429/5xx, timeouts, connection resets.
- Authentication errors are not retried.
- Daily quota exhaustion still fails after the bounded retries; it is not
  something retries can fix.

Tests: `tests/test_llm_retry.py`.

## Synthesis latency

- Single-row, SQL-only answers now use a deterministic fast path and skip the
  synthesis LLM round trip. They are already fully grounded in the SQL result.
- Multi-row SQL, RAG, and `needs_both` answers still use grounded synthesis.
- This reduces latency for the most common aggregate questions without
  weakening citation verification.

Tests: `tests/test_synthesis.py`.

## Eval token budget

`eval/full_eval.py` now supports:

```bash
python -m eval.full_eval --skip-sql-suite   # skip the 15-question SQL suite
python -m eval.full_eval --skip-judge       # skip LLM faithfulness judging
```

The report records `answer_faithfulness.skipped` separately from judge errors,
so partial runs are not mistaken for failures.

## Verification

- `pytest`: **71/71 passed**
- App health: `app`, `database`, `chroma` all `ok`
- SQL safety tests include timeout, row cap, star rejection, and function
  allowlist bypass cases.

## Still open

- Full RBAC, SSO, tenant isolation, and row-level security.
- Immutable audit storage and retention policy.
- TLS termination, secrets manager, and encrypted backups.
- A valid full eval on the configured `gpt-oss-20b` model once quota resets.
- Real embeddings + cross-encoder reranker.
- Frontend and human-in-the-loop review for conflicts.
