# Embeddings

Date: 2026-11-17

## Current choice

LedgerMind now uses **`BAAI/bge-small-en-v1.5`** for dense retrieval:

- 384 dimensions, 512-token context;
- runs locally on the M2 CPU (about 130 MB of weights);
- strong English retrieval for policies, MSAs, and finance questions;
- BGE query instruction:
  `Represent this sentence for searching relevant passages: `;
- document side has no instruction;
- embeddings are L2-normalized, so cosine similarity is valid.

The previous `HashEmbeddingFunction` remains available as a deterministic
offline/test fallback. It is not a semantic model and should not be used for
production retrieval.

## Configuration

```bash
EMBEDDING_BACKEND=sentence_transformers   # hash for offline/tests
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DEVICE=cpu                      # mps may help larger models
EMBEDDING_BATCH_SIZE=32
```

`app/rag/store.py` includes the embedding function name in the Chroma
collection name, e.g.:

```
ledgermind_docs_<tenant>_st-BAAI-bge-small-en-v1.5
```

Changing the model therefore creates a new collection instead of mixing
vectors with incompatible dimensions.

## Prefix handling

`app/rag/embeddings.py` defines a profile per model family:

| Model family | Query prefix | Document prefix |
|---|---|---|
| BGE v1.5 | `Represent this sentence for searching relevant passages: ` | none |
| E5 | `query: ` | `passage: ` |
| Nomic | `search_query: ` | `search_document: ` |
| Qwen3-Embedding | instruction + `Query: ` | none |
| MiniLM / MPNet / multilingual MiniLM | none | none |

`retrieve()` uses `embed_query()` and passes `query_embeddings` to Chroma, so
query instructions are not accidentally applied to stored documents.
Ingestion uses `embed_documents()`.

## Reindexing

After changing the model, re-ingest the corpus:

```bash
cd /Users/ayushikashyap07/Documents/ledgermind
.venv/bin/python -m eval.rag_eval
```

`eval.rag_eval` resets the current tenant collection, re-ingests the corpus, and
runs the 15 retrieval cases.

## Results

| Check | Result |
|---|---|
| Retrieval recall@3 | **15/15 (100%)** |
| `e28` reimbursement query | `expense_policy.txt` ranked **#1** |
| BGE embedding dimension | 384 |
| Query instruction applied | Yes |
| Hash fallback | Tests/offline only |

Raw artifact: `docs/baseline/rag_eval_bge_small.json`.

## Model alternatives on this machine

| Model | Fit |
|---|---|
| `BAAI/bge-small-en-v1.5` | **Recommended default** (current) |
| `BAAI/bge-base-en-v1.5` | Better quality, ~3× slower; worth evaluating next |
| `sentence-transformers/all-MiniLM-L6-v2` | Fastest, lower quality |
| `intfloat/e5-base-v2` | Strong, but requires query/passage prefixes |
| `nomic-embed-text-v1.5` | Long context/Matryoshka; requires prefixes + `trust_remote_code` |
| `Qwen3-Embedding-0.6B` | Best multilingual option that fits 8 GB; slower |
| `BAAI/bge-large-en-v1.5` | Possible but slow on CPU |
| `Qwen3-Embedding-4B/8B` | Not practical with 8 GB RAM |

## Next

1. Evaluate `bge-base-en-v1.5` against the current BGE-small on the same
   retrieval and full-end-to-end suites.
2. Evaluate a larger BGE reranker against the current MiniLM cross-encoder.
3. Replace the synthetic corpus with real filings and table-heavy invoices.
