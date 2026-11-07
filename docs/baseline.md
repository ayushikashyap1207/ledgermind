# LedgerMind baseline — 2026-11-07

This is the first baseline measured **without** the hardcoded SQL fallback and
**with** route-accuracy and citation-correctness checks. It supersedes the
numbers in `docs/phase6_build_report.md` (15/15 SQL, 10/10 router, 1.35 ms
latency), which were produced with the fallback path available and without
verifying routing or citation sources.

> **Update:** the keyword router was replaced with an LLM classifier on the
> same day. See `docs/router_v2.md` for the before/after numbers and the
> remaining `needs_both` decomposition gap.

## Environment

| Item | Value |
|---|---|
| Date | 2026-11-07 |
| Commit before this change | `28d2761` |
| Python | 3.12.0 (`.venv`) |
| LLM provider / model | Groq / `openai/gpt-oss-20b` (live API key) |
| Structured DB | SQLite `./ledgermind.db`, executed through the read-only URI engine |
| Vector store | Chroma persistent `./chroma_data`, hash embeddings, `docs/rag_corpus` |
| Chunking | 700 chars / 120 overlap (existing collection from prior ingestion) |
| LangSmith tracing | Off |
| Unit tests | **12/12 passed** |
| Golden SQL execution | **15/15 passed** |

## What changed before measuring

1. **Removed the hardcoded SQL fallback.**
   - Deleted `_fallback_generate_sql()` from `app/sql/text_to_sql.py`.
   - Removed the now-unused `sql_fallback_enabled` setting from
     `app/core/config.py`.
   - Removed `SQL_FALLBACK_ENABLED` from `.env` and `.env.example`.
   - Provider failures now propagate and are surfaced as `execution_error`
     instead of being answered by a handwritten lookup table.

2. **Made the eval measure routing and citations.**
   - `eval/full_eval.py` now records `expected_route`, `actual_route`,
     `route_correct`, `sql_status`, `rag_hits`, and citation facts per case.
   - Citation correctness verifies both the citation type (SQL/document) and
     that a document citation points at the **expected source**, not just any
     document.
   - The faithfulness judge now receives the actual retrieved chunk text and
     SQL result rows/columns instead of only citation references
     (`app/router/synthesis.py` exposes them additively).

## Results

| Metric | Result | Notes |
|---|---:|---|
| SQL execution correctness | **12/15 (80.0%)** | Generated SQL compared by result set against golden SQL |
| Routing accuracy | **18/30 (60.0%)** | 10 RAG → SQL errors, 2 both → SQL errors |
| Citation correctness | **14/30 (46.7%)** | Type + expected source; dominated by routing failures |
| Answer faithfulness | **19/30 (63.3%)** | Live LLM judge (`judge_mode=llm`), evidence text provided |
| Retrieval recall@3 | **15/15 (100%)** | Synthetic corpus; each case has one expected source |
| Retrieval precision@3 | **0.3333** | Metric artifact: one relevant source per case, so max is 1/3 |
| End-to-end latency p50 | **4,567 ms** | Real Groq calls |
| End-to-end latency mean | **3,452 ms** | |
| End-to-end latency max | **7,258 ms** | |

Raw reports:

- `docs/baseline/sql_eval.json`
- `docs/baseline/full_eval.json`

## Failure analysis

### Routing — the largest gap (12/30 wrong)

| Expected → actual | Count | Example |
|---|---:|---|
| `needs_rag` → `needs_sql` | 10 | "Who approves emergency purchases over 50k?" |
| `needs_both` → `needs_sql` | 2 | "Top vendor by invoiced amount plus concentration cap?" |

Root cause: `app/router/classifier.py` is a keyword scorer with a SQL-biased
default. `_SQL_HINTS` contains generic finance words such as `quarter`,
`account`, and `invoice`, while `_RAG_HINTS` misses common document phrasings
("signatures", "region", "disputes", "cap", "approves", "late payment fee",
"single-vendor exposure"). Any question with no recognized RAG hint falls
through to `needs_sql`.

### SQL generation — 3/15 wrong

- **q02** — "total overdue invoices" generated `due_date < date('now')` instead
  of `status = 'overdue'`. The system prompt does not define "overdue".
- **q11** — "5 most recent transactions" generated `SELECT *`, returning extra
  columns. This is partly a **false negative** in the strict row-set comparator,
  but it is also the `SELECT *` allowlist bypass: extra columns pass validation.
- **q15** — generated a join with an unqualified `vendor_id`, causing
  `ambiguous column name: vendor_id`. No repair/retry loop exists.

Overall SQL statuses across the 30 E2E cases: 6 `ok`, 13 `not_answerable`,
2 `rejected`, 9 not run (correctly routed RAG cases).

### Citations — 16/30 wrong

- 12 failures cascade directly from routing: a document question routed to SQL
  produces no document citation and usually no SQL citation either.
- 3 `needs_both` cases (`e21`–`e23`) have the correct document citation but no
  SQL citation because SQL generation failed.
- `e28` was routed correctly but retrieved the wrong sources: top-3 contained
  `procurement_policy.txt`, `vendor_msa_excerpt.txt`, and
  `earnings_call_q2_2026.txt`, not `expense_policy.txt`.

### Faithfulness — 19/30 faithful

- 14 cases scored 5/5: correctly routed RAG answers grounded in cited chunks.
- 15 cases scored 1/5: mostly misrouted or no-evidence answers, where the
  response could not be supported by the evidence provided.
- 1 case scored 2/5.

The judge now sees evidence text, so this is a more meaningful number than the
previous heuristic (`"result" in answer`). It is still an LLM proxy, not ground
truth.

### Latency

The previous 1.35 ms p50 was measured without live LLM calls. Real provider
latency is ~3.5 s mean / ~4.6 s p50, which is the number that matters for a
user-facing product.

## Reproduce

```bash
cd /Users/ayushikashyap07/Documents/ledgermind

# Golden queries (no LLM) and unit tests
.venv/bin/python -m eval.sql_eval --golden-only
.venv/bin/python -m pytest -q

# Full baseline artifacts
.venv/bin/python -m eval.sql_eval  > docs/baseline/sql_eval.json
.venv/bin/python -m eval.full_eval > docs/baseline/full_eval.json
```

## Caveats

- This is a **single run** with a non-deterministic model; expect a few points
  of variance on re-run.
- The SQL comparator is exact on row values and column sets. `q11` is a
  column-superset false negative; `q02` and `q15` are genuine model errors.
- Retrieval precision@3 is structurally capped at 1/3 because each test case
  expects exactly one source. Treat recall and citation correctness as the
  primary retrieval signals.
- The corpus is small and synthetic, and embeddings are hash-based. These
  retrieval numbers will not transfer unchanged to real 10-Ks or invoices.
- SQLite is used locally; Docker/Postgres and the containerized Chroma path
  were not exercised in this run.
- No LangSmith traces were captured in this run.

## What to fix first, based on this baseline

1. **Router** — replacing the keyword classifier with a structured-output LLM
   classifier (with confidence and rules as fallback) addresses 12 of the 16
   citation failures and most of the faithfulness gap.
2. **SQL prompt + repair** — define "overdue" and period semantics in the
   prompt, require qualified joins, reject `SELECT *` or expand it to the
   allowlist, and add one execution-guided repair attempt.
3. **Retrieval for real corpora** — real embeddings + hybrid BM25/dense +
   reranker; the `e28` miss is the early warning that hash embeddings will not
   generalize.
4. **Synthesis + abstention** — replace template concatenation with grounded
   synthesis and a verified "insufficient evidence" path.
5. **CI gate** — fail the build when routing, citation, or SQL correctness
   regresses below this baseline.
