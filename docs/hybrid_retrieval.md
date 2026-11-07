# Hybrid retrieval — BM25 + dense + rerank

Date: 2026-11-08

## Problem

The retriever relied on hashed bag-of-words embeddings and MMR. That was enough
for 15/15 synthetic retrieval cases, but it missed `e28`:

> "What is the reimbursement timeline for approved expenses?"

`expense_policy.txt` was ranked fourth, so the answer cited the procurement
policy, vendor MSA, and earnings call instead. Exact policy terms such as
"reimbursement" and "expenses" were being averaged away by the dense vector and
then penalized by MMR's diversity term.

## What changed

- Added `app/rag/bm25.py` — a dependency-free BM25 index over the chunks
  currently stored in Chroma.
  - Lowercasing and tokenization for financial terms (`Net-45`, `FY2026`).
  - A small stopword list so function words do not dominate IDF on a small
    corpus.
  - BM25+ style IDF, `k1=1.5`, `b=0.75`.
- Added `app/rag/hybrid.py`:
  - reciprocal rank fusion (RRF) of dense and BM25 ranked lists;
  - min-max score normalization;
  - MMR selection over the fused, reranked candidates.
- Rewrote `retrieve()` in `app/rag/pipeline.py`:
  1. dense candidates from Chroma;
  2. BM25 candidates over all chunks;
  3. union + RRF;
  4. rerank with `0.4 * RRF + 0.4 * BM25 + 0.2 * dense`;
  5. MMR diversification with `lambda=0.85`.
- The BM25 index is cached and invalidated on `upsert_chunks()` and
  `reset_collection()` so re-ingestion cannot serve a stale index.
- Added `tests/test_rag_hybrid.py` (9 tests). Total suite: **34 tests**.

## Results

### Retrieval eval

| Metric | Before | After |
|---|---:|---:|
| Recall@3 | 15/15 (100%) | **15/15 (100%)** |
| Precision@3 | 0.3333 | 0.3333 |

Precision is structurally capped at 1/3 because each test case expects exactly
one source; it is not a useful ranking signal here.

### `e28` now ranks the right document first

| Rank | Source | Hybrid relevance |
|---:|---|---:|
| 1 | `expense_policy.txt` | 0.8128 |
| 2 | `procurement_policy.txt` | 0.7255 |
| 3 | `vendor_msa_excerpt.txt` | 0.1962 |
| 4 | `earnings_call_q2_2026.txt` | 0.1117 |
| 5 | `risk_committee_memo.txt` | 0.0000 |

### Full 30-case eval — model note

The configured `openai/gpt-oss-20b` model hit Groq's 200k-token daily limit
during repeated baseline runs. To get a valid end-to-end measurement without
waiting for the reset, the full suite was run with
`LLM_MODEL=openai/gpt-oss-120b` (same Groq provider, larger model):

| Metric | Result |
|---|---:|
| Routing accuracy | **30/30 (100%)** |
| Citation correctness | **30/30 (100%)** |
| Answer faithfulness | **27/30 (90.0%)** |
| SQL execution correctness | 11/15 (73.3%) |
| Retrieval recall@3 | 15/15 (100%) |
| Latency p50 / mean / max | 5.39s / 6.69s / 15.87s |

For reference, the last valid `gpt-oss-20b` run before decomposition and hybrid
retrieval had citation correctness **24/30** and faithfulness **26/30**. The
120b run is not an apples-to-apples model comparison, but it confirms the
current pipeline can reach 30/30 citations and fixes the `e28` miss. A 20b
re-run is still pending quota.

Raw artifacts:

- `docs/baseline/rag_eval_hybrid.json`
- `docs/baseline/full_eval_hybrid_gpt120b.json`

## Caveats

- The BM25 index is in-process and rebuilds from all chunks. That is fine for
  this corpus; a production corpus should use OpenSearch, Postgres full-text
  search, or a sparse-vector store with incremental indexing.
- Hash embeddings are still weak. BM25 fixes exact-term misses, but semantic
  paraphrases need real embeddings (for example `text-embedding-3` or a local
  sentence-transformer) plus a cross-encoder reranker.
- SQL correctness is still 11/15. The remaining failures are the "overdue"
  business definition, unqualified joins, and strict row/column comparison.
- Latency remains provider-bound (~5–7 s p50/mean). Classification,
  decomposition, SQL generation, and the judge are separate LLM calls.

## Next

1. Re-run the full suite on `gpt-oss-20b` when its daily quota resets, for an
   apples-to-apples result.
2. Add real embeddings + a cross-encoder reranker.
3. Fix the SQL definition and join failures; add an execution-guided repair
   attempt.
4. Add retry/backoff and a per-run token budget for evals.
