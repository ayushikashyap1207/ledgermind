# LedgerMind

Hybrid enterprise financial agent: answers natural-language questions by
routing between a Text-to-SQL engine (structured ledger data) and a RAG
pipeline (10-Ks, transcripts, invoices), synthesizing one cited answer.

## Status

| Phase | What | Status |
|---|---|---|
| 0 | Scaffolding, config, docker-compose, `/health` | ✅ Built & tested |
| 1 | Text-to-SQL + safety validation layer | ✅ Built & tested |
| 2 | RAG ingestion + retrieval (MMR) | ✅ Built (15/15 top-3 retrieval) |
| 3 | Router + answer synthesis | ✅ Built (10/10 mixed routing+citation) |
| 4 | Eval suite (LangSmith hooks + metrics) | ✅ Built (30-case suite) |
| 5 | API + deployment | ✅ Built (`/query`, `/ingest`, `/ingest/sql`, `/eval/run`) |
| 6 | 14-section doc | ✅ Completed (`docs/phase6_build_report.md`) |

**Current checkpoint:** Full phased build through Phase 6 is implemented.
Containerized Chroma verification is pending local Docker daemon availability;
local PersistentClient mode is fully working.

## What's actually proven so far (numbers, not vibes)

- **Synthetic data:** 6,028 rows seeded (`accounts`: 13, `vendors`: 15,
  `transactions`: 1,800, `invoices`: 600, `general_ledger`: 3,600) — clears
  the 5k+ target and is internally consistent (every transaction has
  balanced double-entry debit/credit ledger legs).
- **SQL safety validator:** 12/12 pytest cases pass
  (`tests/test_sql_safety.py`, `tests/test_executor_readonly.py`,
  `tests/test_health.py`), including 10 distinct malicious/off-schema
  payloads (DROP, UPDATE, DELETE, stacked-statement injection, INSERT,
  off-schema table, `sqlite_master` probe, PRAGMA probe, ATTACH DATABASE,
  off-schema column) — all rejected **before** execution.
- **Defense in depth verified independently at both layers:** layer 1
  (`validate_sql`) rejects malicious SQL by parsing it; layer 2 (the
  read-only connection in `executor.py`) was tested by *deliberately
  skipping* layer 1 and confirming the connection itself still refuses to
  write (`test_connection_itself_refuses_writes_even_if_validator_is_bypassed`).
- **15/15 SQL eval** (`python -m eval.sql_eval`) on this machine after
  fallback handling for missing live provider auth.
- **RAG retrieval acceptance:** 15/15 correct chunk in top-3
  (`python -m eval.rag_eval`).
- **Router acceptance:** 10/10 mixed questions route+citation-correct
  (`python -m eval.router_eval`).
- **Phase 4 suite:** 30 cases with retrieval, SQL correctness,
  faithfulness rubric logs, and latency (`python -m eval.full_eval`).
- **7/7 malicious queries** rejected in the standalone eval script
  (`eval/sql_safety_eval.py`) — exceeds the "at least 5" acceptance bar.

## Running it

### Quick local run (no Docker, what's been tested here)
```bash
pip install -r requirements.txt
cp .env.example .env          # fill in an LLM API key to run full evals
python -m ingestion.seed_synthetic_data
python -m ingestion.ingest_documents --experiment
python -m eval.rag_eval
python -m eval.router_eval
pytest tests/ -v
python -m eval.sql_eval                 # uses live provider when available, otherwise fallback SQL generator
python -m eval.full_eval
python -m eval.sql_safety_eval
uvicorn app.main:app --reload           # then GET /health
```

### Full stack (Docker)
```bash
cp .env.example .env   # set ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY
docker-compose up --build
curl http://localhost:8000/health
```
This switches `DATABASE_URL` to Postgres and stands up the `ledgermind_ro`
read-only role automatically (`docker/init-readonly-role.sql`) — the
production-grade version of the layer-2 defense that SQLite fakes locally
via a `mode=ro` URI connection.

One-command local bootstrap:
```bash
./scripts/bootstrap_and_run.sh
```

## Repo layout
```
app/
  core/       # config.py (pydantic settings, pluggable provider config), llm.py (provider adapters)
  sql/        # models.py, schema_registry.py, safety.py, executor.py, text_to_sql.py, pipeline.py
  rag/        # Phase 2 ingestion/retrieval/MMR
  router/     # Phase 3 route + synthesis pipeline
  main.py     # FastAPI app + query/ingest/eval endpoints
ingestion/    # seed_synthetic_data.py + ingest_documents.py
eval/         # sql, rag, router, and full eval suites
tests/        # pytest — safety validator, executor, health
docker/       # Dockerfile, init-readonly-role.sql
docker-compose.yml
docs/phase6_build_report.md
```

## Design notes for whoever picks this up next
- `app/sql/safety.py` is kept deliberately separate from `text_to_sql.py`
  on purpose — this is the file to point at and say "this is what stops a
  bad query" in an interview.
- The table/column allowlist in `safety.py` is *derived* from the same
  SQLAlchemy models used for the schema (`schema_registry.py`), not
  hand-maintained separately, so it cannot silently drift from the real
  schema.
- Don't add LangGraph for the Phase 3 router unless it turns out to
  genuinely need multi-step state — a classifier + two chains is fine and
  is itself a better interview answer than over-engineering.
