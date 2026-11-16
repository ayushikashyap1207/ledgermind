# Cross-encoder reranker

Date: 2026-11-17

## Current choice

`cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the fused retrieval candidates.

- Lightweight (~90 MB), fast on the M2 CPU.
- Scores each `(question, chunk)` pair directly instead of relying on vector
  similarity.
- About **50 ms per query** after the embedding and reranker models are warm.
- Configurable via environment variables.

## Retrieval flow

1. BGE-small dense candidates from Chroma.
2. BM25 candidates over the tenant's chunks.
3. Reciprocal rank fusion.
4. Hybrid base relevance:
   `0.4 * RRF + 0.4 * BM25 + 0.2 * dense`.
5. Cross-encoder scores every candidate.
6. Final score:
   `0.7 * normalized cross-encoder + 0.3 * hybrid base relevance`.
7. Top-k selected. If the reranker fails or is disabled, the previous
   hybrid + MMR path is used.

## Configuration

```bash
RERANKER_ENABLED=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_DEVICE=cpu
RERANKER_BATCH_SIZE=16
```

Set `RERANKER_ENABLED=false` to fall back to hybrid RRF + MMR. Tests set this
automatically so the suite stays offline and fast.

## Results

| Check | Result |
|---|---|
| Retrieval recall@3 | **15/15 (100%)** |
| `e28` reimbursement query | `expense_policy.txt` ranked **#1** |
| Procurement threshold | `procurement_policy.txt` ranked **#1** |
| Contracts/signatures | `procurement_policy.txt` ranked **#1** |
| Rerank latency (warm) | ~46–50 ms/query |

Raw artifact: `docs/baseline/rag_eval_bge_reranker.json`.

## Why not MMR alone?

MMR diversifies candidates using embedding similarity, but it does not read the
question and passage together. The cross-encoder can distinguish "approval
threshold" from "late payment fee" even when their bag-of-words/vector profiles
overlap. MMR remains the fallback when reranking is disabled.

## Alternatives

| Model | Trade-off |
|---|---|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | **Current:** fast, low memory |
| `BAAI/bge-reranker-base` | Stronger, ~1.1 GB; slower on CPU |
| `BAAI/bge-reranker-v2-m3` | Best multilingual quality; too heavy for routine local use |
| `jinaai/jina-reranker-v2-base-multilingual` | Strong multilingual option; larger |

On this 8 GB M2, the MiniLM cross-encoder is the best quality/latency trade-off.
Evaluate the BGE reranker on a larger real-document set before switching.

## Next

1. Evaluate the MiniLM reranker vs `BAAI/bge-reranker-base` on real filings.
2. Add table-aware chunking so reranker passages contain complete tables.
3. Add a retrieval regression gate to CI.
