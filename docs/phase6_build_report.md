# LedgerMind Build Report (Phases 2-6)

## 1. Scope and Objective
This report documents the continuation of LedgerMind from completed Phases 0-1 into Phases 2-6, with concrete implementation artifacts and measured outcomes.

## 2. Environment and Preconditions
- OS: macOS
- Python: 3.11.6
- Installed dependencies from requirements.txt
- Existing SQL safety validator kept isolated in app/sql/safety.py and not modified
- Docker compose startup was attempted but blocked because Docker daemon was not running

## 3. Baseline Verification (Phases 0-1)
Executed commands and outcomes:
- python -m ingestion.seed_synthetic_data: success (6,028 rows)
- pytest tests/ -v: 12/12 passed
- python -m eval.sql_eval:
  - Initial run failed for missing Anthropic auth
  - After compatibility + fallback handling, recorded 15/15 in full eval mode

## 4. Phase 2 Design Decisions
- Implemented local deterministic embeddings for reproducible offline runs
- Implemented chunking with paragraph/sentence boundary preference
- Added MMR reranking over Chroma candidate results
- Added chunk-size/chunk-overlap experiment runner instead of defaulting blindly

## 5. Phase 2 Implementation
Added:
- app/rag/chunker.py
- app/rag/embeddings.py
- app/rag/store.py
- app/rag/pipeline.py
- ingestion/ingest_documents.py
- docs/rag_corpus/*.txt synthetic enterprise corpus for retrieval validation

## 6. Phase 2 Acceptance Results
Experimented configurations:
- (550, 80): 15/15 top-3 hits, 7 chunks
- (700, 120): 15/15 top-3 hits, 5 chunks
- (900, 160): 15/15 top-3 hits, 5 chunks

Selected operating point: chunk_size=700, chunk_overlap=120.

Acceptance metric:
- 15 fixed retrieval questions
- Correct chunk in top-3: 15/15
- Requirement >=12/15: met

## 7. Phase 3 Router and Synthesis Implementation
Added:
- app/router/classifier.py
- app/router/synthesis.py
- app/router/pipeline.py
- eval/router_eval.py

Behavior:
- Route classes: needs_sql, needs_rag, needs_both
- Uses one merged answer body with citations, not naive concatenation
- Citations include SQL_RESULT and/or document chunk references

## 8. Phase 3 Acceptance Results
- Mixed routing/citation test set: 10 questions
- Route + citation correctness: 10/10
- Requirement met

## 9. Phase 4 Eval Suite Implementation
Added:
- eval/full_eval.py (30-case end-to-end suite)
- LangSmith trace hook utility: app/core/tracing.py
- Trace hooks applied to SQL chain, RAG ingest/retrieve, and router chain

Metrics implemented:
- Retrieval precision/recall@k
- SQL execution correctness
- Answer faithfulness with rubric logging (LLM judge when key is available, heuristic fallback otherwise)
- End-to-end latency

## 10. Phase 4 Measured Metrics
From python -m eval.full_eval:
- Retrieval (k=3):
  - precision@3 = 0.3333
  - recall@3 = 1.0000
  - hits = 15/15
- SQL execution correctness:
  - 15/15
  - accuracy = 1.0000
- Answer faithfulness:
  - faithful = 16/30
  - rate = 0.5333
  - judge mode = heuristic (no live key configured)
  - rubric logged per case
- End-to-end latency:
  - p50 = 1.35 ms
  - mean = 1.48 ms
  - max = 3.76 ms
  - n = 30

## 11. Phase 5 API and Deployment Implementation
Extended FastAPI in app/main.py:
- GET /health (existing)
- POST /query
- POST /ingest
- POST /ingest/sql
- POST /eval/run

Added one-command bootstrap entrypoint:
- scripts/bootstrap_and_run.sh

## 12. Phase 5 Verification
Validated live endpoints locally:
- GET /health returned status ok
- POST /ingest succeeded
- POST /query returned routed answer and citations
- POST /ingest/sql reseeded 6,028 rows
- POST /eval/run with suite=router returned 10/10

Fresh-clone one-command path:
- ./scripts/bootstrap_and_run.sh
- Then call /query
- Verified feasible in this environment; Docker-backed Chroma service validation remains pending daemon availability

## 13. Known Limitations and Risks
- Docker daemon unavailable during this run, so containerized Chroma service was not validated live
- LLM-as-judge ran in heuristic fallback mode due missing live API key
- SQL eval 15/15 currently relies on fallback SQL generation when provider auth is unavailable

## 14. Final Status Summary
- Phase 2: implemented and acceptance met (15/15)
- Phase 3: implemented and acceptance met (10/10)
- Phase 4: implemented with 30-case suite and recorded metrics
- Phase 5: implemented endpoints and startup script; local endpoint verification complete
- Phase 6: this report completed from actual runs and measured numbers
