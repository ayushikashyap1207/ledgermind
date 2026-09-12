# LedgerMind

**A hybrid enterprise financial agent that answers natural-language questions by reasoning over both unstructured documents (10-Ks, earnings calls, invoices, policy docs) and structured data (a transactional ledger/database) — and returns one grounded, cited answer.**

![LedgerMind API Docs](docs/images/ledgermind-api.png)
*The FastAPI/Swagger interface for LedgerMind's `/query`, `/ingest`, `/ingest/sql`, `/eval/run`, and `/health` endpoints.*

---

## Table of Contents

- [What is LedgerMind?](#what-is-ledgermind)
- [Why this exists](#why-this-exists)
- [Design principles](#design-principles)
- [Architecture](#architecture)
- [Tech stack — and why](#tech-stack--and-why)
- [How it works](#how-it-works)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Build plan / phases](#build-plan--phases)
- [Evaluation](#evaluation)
- [Getting started](#getting-started)
- [Limitations & known failure modes](#limitations--known-failure-modes)
- [Roadmap](#roadmap)

---

## What is LedgerMind?

Most "financial AI" demos pick a lane: either a RAG chatbot over PDFs, or a text-to-SQL tool over a database. Real finance questions don't respect that split. A question like *"How does Q3 revenue in the 10-K filing compare to what's actually posted in the ledger?"* needs **both** — one number from a document, one number from a database, reconciled into a single, traceable answer.

LedgerMind is that system. A **router** looks at each incoming question and decides whether it needs:

- the **RAG pipeline** (unstructured documents),
- the **Text-to-SQL engine** (structured data), or
- **both**, in which case it fans out to each pipeline and an **answer synthesizer** merges the results into one coherent response — never just two answers stapled together.

Every claim in the final answer is traceable back to either a specific document chunk or a specific SQL query/result set.

## Why this exists

This project is built to be *measured*, not just demoed. It's easy to build something that returns plausible-looking answers; it's harder to prove those answers are correct, safe, and grounded. LedgerMind treats retrieval quality, SQL correctness, and answer faithfulness as first-class outputs — every phase ships with a number attached, not a vibe.

## Design principles

These four rules are non-negotiable and shape every design decision below:

1. **Text-to-SQL is read-only and validated.** No generated query can ever mutate data, and every query is checked *before* it touches the database — never after.
2. **Every answer is traceable.** Each claim cites the document chunk or SQL query/rows that produced it.
3. **The LLM provider is pluggable.** Swapping Anthropic ↔ OpenAI ↔ Groq is a config change, never a code change.
4. **Nothing ships without an eval number.** "It works on my test question" is not a completion criterion anywhere in this project.

## Architecture

```
                        ┌─────────────────┐
   User Question ──────▶│  Router/Agent    │
                        └───────┬─────────┘
                   ┌────────────┼─────────────┐
                   ▼                          ▼
          ┌────────────────┐         ┌──────────────────┐
          │  RAG Pipeline   │         │  Text-to-SQL      │
          │  (ChromaDB +    │         │  Engine           │
          │   MMR rerank)   │         │  (schema-aware +  │
          └────────────────┘         │   safety-validated)│
                   │                          │
                   └────────────┬─────────────┘
                                 ▼
                     ┌────────────────────┐
                     │  Answer Synthesizer │
                     │  (merges + cites)   │
                     └────────────────────┘
                                 │
                                 ▼
                          Grounded Answer
                        + citations/sources
```

The router classifies each question as `needs_sql`, `needs_rag`, or `needs_both`. For `needs_both` questions, both pipelines run and their outputs are merged by the synthesizer rather than concatenated — the goal is one coherent answer, not two answers side by side.

## Tech stack — and why

| Layer | Choice | Why |
|---|---|---|
| Language | **Python 3.11+** | The de facto standard for LLM orchestration, data tooling, and ML libraries; keeps the whole stack in one language. |
| API | **FastAPI** | Async-first, automatic OpenAPI/Swagger docs out of the box (the screenshot above is FastAPI's built-in `/docs` UI), and typed request/response models via Pydantic. |
| Structured data | **PostgreSQL** (SQLite for local dev) | A real, production-grade RDBMS so the Text-to-SQL layer is tested against realistic constraints — foreign keys, types, a read-only role — not a toy in-memory table. |
| Vector store | **ChromaDB** | Lightweight, embeddable, and easy to run as its own Docker service — no need for a hosted vector DB to prototype a real RAG pipeline. |
| Orchestration | **LangChain** (chains) | Handles the prompt templates, LLM calls, and retrieval glue. LangGraph is deliberately *not* used by default — it's only added if the router genuinely needs multi-step state or backtracking, since a classifier + two chains is a perfectly good router on its own. |
| Evaluation & tracing | **LangSmith** | Every chain (router, SQL generation, retrieval, synthesis) is traced so a run can be inspected step by step, not just judged by its final output. |
| LLM providers | **Anthropic + OpenAI + Groq**, behind one interface | No vendor lock-in — the provider is a config value, not a code path, so the system can be benchmarked or swapped across models. |
| Containerization | **Docker + docker-compose** | The app, Postgres, and Chroma run as separate services so the whole system reproduces identically from a fresh clone. |
| Testing | **pytest** | Standard, and used specifically to unit-test the SQL safety-validation layer in isolation from SQL generation. |

## How it works

1. **Ingestion.** Documents (10-K-style filings, transcripts, invoices, policy docs) are chunked, embedded, and stored in ChromaDB. Structured data lives in Postgres across an `accounts` / `transactions` / `invoices` / `general_ledger` / `vendors` schema.
2. **Routing.** An incoming question is classified as `needs_sql`, `needs_rag`, or `needs_both`.
3. **Retrieval / SQL generation.**
   - RAG: ChromaDB retrieval with MMR re-ranking for diverse, non-redundant context.
   - Text-to-SQL: the LLM generates a query against the known schema, which then passes through a dedicated **safety-validation module** before it's ever executed:
     - only `SELECT` statements are allowed — anything else is rejected outright,
     - a table/column allowlist derived from the schema is enforced,
     - every query has a row-limit cap,
     - and the database connection itself uses a **read-only Postgres role** as defense in depth, so the safety net doesn't depend solely on the LLM behaving.
4. **Synthesis.** Results from whichever pipeline(s) ran are merged into a single answer, with every claim tagged to its source — a document chunk or a SQL result set.

## API reference

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Health check for the app, Postgres, and Chroma. |
| `POST` | `/query` | Ask a natural-language question; returns a grounded, cited answer. |
| `POST` | `/ingest` | Ingest unstructured documents into the RAG pipeline. |
| `POST` | `/ingest/sql` | Bulk-load structured data into the ledger schema. |
| `POST` | `/eval/run` | Run the evaluation suite and produce a metrics report. |

Full interactive documentation (as shown in the screenshot) is generated automatically by FastAPI and served at `/docs`, with the raw OpenAPI schema at `/openapi.json`.

## Project structure

```
/app          FastAPI service — router, RAG pipeline, Text-to-SQL engine, synthesizer
/ingestion    Document and SQL data ingestion pipelines
/eval         Golden dataset, evaluation harness, LangSmith wiring
/docs         Project documentation, diagrams, screenshots
/tests        pytest suite (SQL safety validation has its own dedicated tests)
/docker       docker-compose.yml and service configs
```

## Build plan / phases

The system was built in order, with each phase gated on a measured acceptance criterion — not skipped ahead of:

| Phase | Focus | Acceptance criterion |
|---|---|---|
| 0 | Scaffolding & Docker | `docker-compose up` boots app + Postgres + Chroma; `/health` returns 200 from all three. |
| 1 | Text-to-SQL + safety layer | 15/15 test questions produce correct SQL and results; 5+ malicious/off-schema questions are rejected pre-execution. |
| 2 | RAG pipeline | Correct source chunk appears in the top-3 results for at least 12/15 test questions. |
| 3 | Router + synthesis | 10 mixed SQL/RAG/both questions route and cite correctly. |
| 4 | Evaluation suite | Golden set of 30–50 Q/A pairs; retrieval precision/recall, SQL correctness, answer faithfulness, and latency all logged with a LangSmith trace per question. |
| 5 | API + deployment | Fresh clone → one command → working `/query` in under 5 minutes. |
| 6 | Documentation | 14-section technical spec, written last, from actual measured results. |

## Evaluation

LedgerMind treats evaluation as a deliverable, not an afterthought. The Phase 4 suite computes and logs, per run:

- **Retrieval precision/recall @k** — is the RAG pipeline finding the right chunks?
- **SQL execution correctness** — does the generated query return the *right rows*, not just syntactically valid SQL?
- **Answer faithfulness** — does every claim in the final answer trace back to retrieved evidence (scored via an LLM-as-judge rubric, with the rubric itself logged for auditability)?
- **End-to-end latency** — how long does a full round trip take?

Every golden-set question produces both a JSON metrics report and a linked LangSmith trace, so any run can be inspected step by step.

## Getting started

```bash
git clone <repo-url>
cd ledgermind
cp .env.example .env        # set LLM provider + API keys
docker-compose up
```

Once the stack is healthy (`GET /health` returns 200 from the app, Postgres, and Chroma), try:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How does Q3 revenue in the 10-K compare to the general ledger?"}'
```

## Limitations & known failure modes

- Text-to-SQL accuracy degrades on schema questions outside the allowlisted tables/columns — these are rejected by design rather than answered incorrectly.
- RAG retrieval quality is bounded by chunking strategy; very long, table-heavy filing sections are harder to retrieve precisely.
- LLM-as-judge faithfulness scoring is a proxy, not a ground truth — it's logged alongside its rubric so it can be audited, not trusted blindly.
- Multi-hop `needs_both` questions are the hardest case for the synthesizer and have the widest variance in answer quality.

## Roadmap

- Expand the golden evaluation set beyond 50 questions as more document types are ingested.
- Revisit LangGraph if/when the router needs genuine multi-turn state or backtracking (deliberately deferred — see [Design principles](#design-principles)).
- Cost/latency analysis across the three pluggable LLM providers.

---

*Built with FastAPI, LangChain, ChromaDB, PostgreSQL, and LangSmith.*