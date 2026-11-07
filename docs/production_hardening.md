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

## Identity, roles, tenants, and audit

- `app/core/identity.py` adds `tenants`, `users`, `api_keys`, and
  `audit_events` tables with a separate SQLAlchemy Base so they never enter the
  Text-to-SQL allowlist.
- API keys are stored as SHA-256 hashes with a shown-once raw token, active
  flag, and revocation endpoint.
- Roles are enforced per endpoint: viewer (query), analyst (+ingest),
  controller (+eval), admin (+tenant/key management).
- Document retrieval is tenant-scoped through per-tenant Chroma collections.
- SQL execution is fail-closed: only the default tenant can use the shared
  ledger; creating a non-default tenant with `sql_enabled=true` is rejected.
- Audit middleware writes tenant/user/role/request metadata and a question
  hash to `audit_events`, never question text, documents, SQL, or keys.
- Anonymous access is rejected once any active database key exists, even if
  `AUTH_REQUIRED` was forgotten.

See `docs/identity_and_tenants.md` for the model, role table, bootstrap flow,
and remaining gaps.

Tests: `tests/test_auth.py`, `tests/test_identity.py`.

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

- `pytest`: **81/81 passed**
- App health: `app`, `database`, `chroma` all `ok`
- SQL safety tests include timeout, row cap, star rejection, and function
  allowlist bypass cases.

## Still open

- SSO/OIDC; the current keys are manually issued.
- A dedicated database/schema per tenant and Postgres row-level security.
- Alembic migrations; `create_all()` is local/test bootstrap only.
- Immutable audit storage, retention policy, and SIEM export.
- TLS termination, secrets manager, and encrypted backups.
- A valid full eval on the configured `gpt-oss-20b` model once quota resets.
- Real embeddings + cross-encoder reranker.
- Frontend and human-in-the-loop review for conflicts.
