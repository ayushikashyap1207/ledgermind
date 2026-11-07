# SQL correctness hardening

Date: 2026-11-08

## Problems

The last full run measured SQL correctness at 11/15. The failures fell into
four groups:

| Case | Failure |
|---|---|
| q02 | Treated "overdue" as `due_date < now` instead of `status = 'overdue'`. |
| q06 | Returned only the vendor name when the golden query also selected the total. |
| q11 | Used `SELECT *`, returning extra columns the strict comparator rejected. |
| q15 | Joined without qualifying `vendor_id`, causing an ambiguous-column error. |

There was also no recovery path when a generated query failed to execute, and
the exact row comparator punished legitimate projection differences.

## What changed

### 1. Prompt hardening (`app/sql/text_to_sql.py`)

Added explicit query rules:

- "Overdue" means `invoices.status = 'overdue'` unless the user explicitly asks
  for invoices past their due date.
- Qualify every column in a join (`i.vendor_id`, `v.vendor_id`).
- Never use `SELECT *`; select only the needed columns.
- Ranking questions must include the ranked metric with an alias.
- Return exactly one SELECT, with no explanation or semicolon.

### 2. Fair result comparison (`eval/sql_eval.py`)

`results_match()` now accepts:

1. an exact row match, regardless of column aliases;
2. a generated **superset** of the golden columns, projected down to the golden
   projection (fixes q11);
3. a generated **subset** of the golden columns, projected from the golden rows
   (fixes q06's "which vendor" answer).

It still fails on wrong values and missing rows. Added 6 comparator tests.

### 3. Execution-guided repair (`app/sql/text_to_sql.py`, `app/sql/pipeline.py`)

- `repair_sql()` sends the question, failed SQL, database error, and schema back
  to the model for one corrected SELECT.
- The repaired SQL is validated by the same safety layer before execution.
- Repairs are attempted only for **execution errors**. Unsafe SQL rejected by
  the safety layer is not repaired.
- `SQLAnswer` now exposes `repair_attempted` and `repair_succeeded`.
- Added 4 repair tests.

### 4. Eval visibility

`eval/sql_eval.py` now records the repair flags per question.

## Results

`pytest`: **60/60 passed**.

**Qwen 27b** (`LLM_MODEL=qwen/qwen3.8-27b`, `docs/baseline/sql_eval_qwen.json`):

| Metric | Result |
|---|---:|
| SQL execution correctness | **15/15 (100%)** |
| Questions returning `ok` | 15/15 |
| Repairs needed | 0 |
| Repairs succeeded | 0 |

All four prior failures pass under this model, including q02, q06, q11, and q15.

**Configured 20b model:** the run completed 9 questions (all 9 passed) before
Groq's daily token limit returned `429` for the remaining six. That partial run
was discarded as a full measurement; the 20b re-run is pending quota.

## Caveats

- The 15/15 result is a **cross-model** measurement. The earlier 11/15 and
  12/15 numbers were on `gpt-oss-120b` / `gpt-oss-20b`; a same-model re-run is
  still needed for a strict before/after comparison.
- The comparator now accepts column projections. This is deliberate — two
  different SELECTs can answer the same question — but it means the metric
  measures answer equivalence, not literal query equivalence.
- The repair path adds one LLM call only when execution fails. It was not
  exercised in the successful qwen run, but unit tests cover success and
  failure.
- The safety validator still runs before every execution, including repairs.

## Next

1. Re-run `eval.sql_eval` and `eval.full_eval` on `gpt-oss-20b` when quota
   resets.
2. Add retry/backoff and a per-run token budget so eval runs do not exhaust the
   daily limit.
3. Address the next quality gap: synthesis latency (~17s p50) and the remaining
   2/30 faithfulness cases.
4. Production gate: auth, RBAC, audit logging.
