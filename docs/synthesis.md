# Grounded answer synthesis

Date: 2026-11-08

## Problem

The previous synthesizer was template-based. It pasted the first SQL rows and
the first 260 characters of the top document chunk. That caused the three
remaining faithfulness failures in the last full run:

- `e01` quoted the wrong policy section instead of the sentence containing the
  25,000 USD procurement threshold;
- `e02` cut off mid-sentence before "two signatures";
- `e25` included the SQL total but dropped the audit-exception cap.

The system claimed to produce "one coherent, cited answer" but was actually
showing context snippets.

## What changed

`app/router/synthesis.py` now implements a grounded synthesis pipeline:

1. Build numbered evidence:
   - `D1..Dn` for document chunks, including source and full text;
   - `S1` for the SQL result, including executed SQL, columns, rows, and count.
2. Ask the LLM for one direct answer plus claim-level citations:
   ```json
   {
     "answer": "... [D1] ...",
     "claims": [{"text": "...", "citations": ["D1"]}],
     "insufficient_evidence": false,
     "missing_evidence": "",
     "conflicts": []
   }
   ```
3. Verify the response:
   - citation IDs must exist in the evidence;
   - every claim must cite at least one source;
   - every claim must appear in the answer;
   - all numbers in a claim must appear in the sources **that claim cites**,
     not merely somewhere else in the evidence;
   - when both document and SQL evidence are present, both must be cited;
   - the answer must contain inline `[D#]`/`[S#]` markers;
   - numeric scale words are normalized (`18 million` matches `18,000,000`);
   - an `insufficient_evidence` response must say what is missing.
4. If verification fails, retry once with the exact problems.
5. If the retry fails or the provider is unavailable, fall back to a
   deterministic extractive answer that selects the most query-relevant full
   sentences and formats the SQL result.

Additional changes:

- `POST /query` now returns `answer_method`, `claims`,
  `insufficient_evidence`, and `conflicts`.
- Citations carry an `evidence_id` so claim references can be resolved.
- Eval rows record `answer_method` and `insufficient_evidence`.
- The faithfulness judge now receives the same `[D#]`/`[S#]` evidence IDs the
  answer cites, instead of raw file references.
- Added an **Ollama provider** (`LLM_PROVIDER=ollama`) for local/self-hosted
  runs, useful when hosted quotas are exhausted.

## Validation

- `pytest`: **50/50 passed**, including 15 synthesis tests.
- Local Ollama validation:
  - `e01` initially cited the wrong document; per-claim verification caught it,
    and the repaired answer cited `[D1]` correctly.
  - `e30` answered "18 million USD" from the risk memo correctly.
  - `e28` answered "reimbursed within 10 business days" from the expense
    policy, which the old synthesizer missed.
- Valid full run with `openai/gpt-oss-120b`
  (`docs/baseline/full_eval_synthesis_gpt120b.json`):

| Metric | Before synthesis | After synthesis |
|---|---:|---:|
| Routing accuracy | 30/30 | **30/30** |
| Citation correctness | 30/30 | **30/30** |
| Answer faithfulness | 27/30 | **28/30 (93.3%)** |
| SQL correctness | 11/15 | 11/15 |
| Latency p50 | 5.39s | 17.23s |
| Answer methods | extractive only | 25 LLM / 5 extractive |

The two faithfulness failures (`e21`, `e22`) were caused by the judge receiving
file references instead of evidence IDs. That bug is fixed. A partial post-fix
120b run judged **22/22 faithful** before the daily token limit interrupted it;
the full post-fix run is pending quota.

## Trade-off: latency

Synthesis adds one LLM call per query, plus occasional verification retries, so
p50 rose from ~5.4s to ~17.2s on the 120b model. This is the expected cost of
replacing snippets with verified answers.

Next optimizations:

- run synthesis on a smaller/faster model than routing and SQL;
- cache repeated questions and common evidence sets;
- stream the answer;
- reduce retry rate with a tighter prompt;
- only send the top 1–2 chunks to synthesis when the route is `needs_both`.

## Caveats

- The full post-fix run is blocked by Groq's 200k-token daily limit. The
  120b result is a cross-model comparison against the 20b baseline, not an
  apples-to-apples run.
- The local `llama3.2:3b` model is useful for plumbing checks but too weak for
  production routing/SQL; it is not used by default.
- `conflicts` are surfaced but not yet used to block an answer. A finance
  workflow should require human review when a document and the ledger disagree.
- The extractive fallback is grounded but less fluent than the LLM path.

## Next

1. Re-run the full suite on the configured 20b model once quota resets.
2. Add retry/backoff and a per-run token budget.
3. Fix the remaining SQL failures (overdue definition, join qualification,
   projection strictness).
4. Use synthesis latency measurements to choose a production model.
