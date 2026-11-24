# Optional AWS integration

Date: 2026-11-24

LedgerMind runs locally by default. AWS is an optional adapter layer, not a
requirement for the core RAG pipeline.

Included integrations:

- **Amazon Bedrock** as an LLM provider.
- **Bedrock Guardrails** attached to every Bedrock Converse call.
- **Amazon S3** as a document-ingestion source.

## 1. Bedrock LLM provider

Install the optional AWS dependencies:

```bash
pip install -r requirements-aws.txt
```

Configure:

```bash
LLM_PROVIDER=bedrock
LLM_MODEL=anthropic.claude-3-5-sonnet-20241022-v2:0
AWS_REGION=us-east-1
AWS_PROFILE=                # optional; otherwise the default credential chain
```

Enable model access for the selected model in the Bedrock console. The app uses
the standard boto3 credential chain (environment, profile, SSO, or task role),
so no AWS keys live in the repository or `.env`.

Implementation: `app/core/llm.py::BedrockClient` calls
`bedrock-runtime.converse` with the system prompt, user message, max tokens, and
temperature. The rest of LedgerMind (router, SQL generation, decomposition,
synthesis, judge) uses it through the existing `LLMClient` interface.

## 2. Bedrock Guardrails

Guardrails are applied to Bedrock Converse responses:

```bash
BEDROCK_GUARDRAIL_ENABLED=true
BEDROCK_GUARDRAIL_ID=your-guardrail-id
BEDROCK_GUARDRAIL_VERSION=1
```

If Bedrock returns `stopReason: guardrail_intervened`, the client raises
`GuardrailViolation`. The synthesizer catches it and returns a safe refusal:

> I can't provide an answer because the safety policy blocked it.

`answer_method` is set to `guardrail`, and `insufficient_evidence` is true.
Other LLM stages fall back to their deterministic behavior.

Guardrails require the Bedrock provider. They are not applied to local
Ollama/Groq/OpenAI calls.

IAM permissions:

```json
{
  "Effect": "Allow",
  "Action": [
    "bedrock:InvokeModel",
    "bedrock:ApplyGuardrail"
  ],
  "Resource": "*"
}
```

Restrict the resource to the approved model and guardrail ARNs in production.

## 3. S3 document ingestion

Endpoint:

```http
POST /ingest/s3
Authorization: Bearer <analyst-or-higher-key>
Content-Type: application/json

{
  "bucket": "ledgermind-docs",
  "prefix": "tenant-a/policies/",
  "reset": false
}
```

If `bucket` is omitted, `S3_BUCKET` is used. `prefix` defaults to `S3_PREFIX`.

Supported documents: `.txt`, `.md`, `.pdf`. PDFs are extracted with `pypdf`;
Textract/OCR is not implemented yet.

Each chunk stores:

- `source`: `s3://<bucket>/<key>`
- `source_type`: `s3`
- `s3_bucket`, `s3_key`, `s3_etag`, `s3_version_id`
- tenant ID

This means citations in answers point back to the exact S3 object.

IAM permissions:

```json
{
  "Effect": "Allow",
  "Action": ["s3:ListBucket"],
  "Resource": "arn:aws:s3:::ledgermind-docs"
},
{
  "Effect": "Allow",
  "Action": ["s3:GetObject"],
  "Resource": "arn:aws:s3:::ledgermind-docs/*"
}
```

The bucket should be private, encrypted with SSE-KMS, and versioned.

## Configuration summary

```bash
AWS_REGION=us-east-1
AWS_PROFILE=
AWS_ENDPOINT_URL=
BEDROCK_GUARDRAIL_ENABLED=false
BEDROCK_GUARDRAIL_ID=
BEDROCK_GUARDRAIL_VERSION=DRAFT
S3_BUCKET=
S3_PREFIX=
S3_MAX_OBJECT_BYTES=20000000
```

## Testing

The AWS adapters are tested with fake boto3 clients, so the suite needs no AWS
credentials:

- `tests/test_bedrock.py` — Converse output, guardrail config, intervention,
  safe synthesis refusal.
- `tests/test_s3_ingest.py` — listing, filtering, version metadata, tenant
  scoping.

Total suite: **98 tests**.

## What is intentionally not included

- No S3 event/SQS ingestion worker yet; `/ingest/s3` is synchronous.
- No Textract table/OCR extraction yet.
- No Cognito; the existing tenant/RBAC model remains.
- No ECS/Terraform deployment; the core project stays local/Docker-first.

Those are separate production-platform concerns, not required for the AI/RAG
demo.
