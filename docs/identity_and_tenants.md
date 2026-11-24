# Identity, tenants, and audit

Date: 2026-11-08

This layer turns the single shared bearer token into a real user/tenant model.

## Data model

`app/core/identity.py` defines a separate SQLAlchemy `IdentityBase` so these
tables never leak into the Text-to-SQL schema allowlist.

| Table | Purpose |
|---|---|
| `tenants` | Tenant ID, display name, and whether shared SQL is allowed |
| `users` | Tenant-scoped user email and role |
| `api_keys` | SHA-256 hash, prefix, active flag, last-used timestamp |
| `audit_events` | Append-only request metadata and question hash |

Raw API keys are shown **once** when created. Only the SHA-256 hash is stored.

## Roles

| Role | Rank | Can do |
|---|---:|---|
| `viewer` | 1 | `/query` |
| `analyst` | 2 | `/query`, `/ingest` |
| `controller` | 3 | Analyst actions + `/eval/run` (default tenant only) |
| `admin` | 4 | Everything, including tenant and API-key management |

## Endpoint protection

- `/query` — viewer+
- `/ingest` — analyst+
- `/ingest/sql` — admin, and only for a tenant with SQL enabled
- `/eval/run` — controller+, default tenant only
- `/admin/tenants` — admin
- `/admin/api-keys` — admin
- `/admin/api-keys/{key_id}/revoke` — admin
- `/health`, `/docs` — public

## Tenant isolation

- **Documents:** each tenant has its own Chroma collection
  (`ledgermind_docs_<tenant>`), and ingestion/retrieval are tenant-scoped.
- **SQL:** only the default tenant may use the shared ledger database. Creating
  a non-default tenant with `sql_enabled=true` is rejected with `400`.
  Non-default tenants get RAG answers only until they are given a dedicated,
  isolated database connection.
- **Audit:** every event is tagged with tenant, user, role, and request ID.

This is deliberately fail-closed: a missing isolation mechanism denies access
instead of falling back to the shared ledger.

## Authentication modes

1. **Local development:** with no `API_KEY`, no `AUTH_REQUIRED`, and no active
   database keys, requests use a default local admin context.
2. **Bootstrap key:** set `API_KEY` and `AUTH_REQUIRED=true`. The legacy key is
   compared timing-safely and acts as a default-tenant admin.
3. **Database keys:** use the bootstrap admin to create tenants and per-user
   keys:
   ```bash
   curl -X POST http://localhost:8000/admin/tenants \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"tenant_id":"acme","name":"Acme Finance"}'

   curl -X POST http://localhost:8000/admin/api-keys \
     -H "Authorization: Bearer $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"tenant_id":"acme","email":"analyst@acme.com","role":"analyst"}'
   ```
   The response contains the raw key once.

Once any active database key exists, anonymous requests are rejected even if
`AUTH_REQUIRED` was forgotten. Key revocation sets `active=false`; the next
request using that key gets `401`.

## Audit trail

`AuditMiddleware` writes to `audit_events` and emits a structured log line:

- tenant, user, role, request ID, method, path, status, duration, client;
- whether the call was authenticated;
- a SHA-256 hash of the question, never the question text;
- never document content, SQL text, or API keys.

The application has no update or delete endpoint for `audit_events`. A
production deployment should also grant the app role `INSERT`/`SELECT` only on
that table and ship events to a write-once store or SIEM.

## Tests

- `tests/test_auth.py` — legacy/disabled auth and `401` behavior
- `tests/test_identity.py` — DB keys, roles, revocation, tenant-scoped
  retrieval, admin endpoints
- `tests/test_health.py` — audit middleware and request IDs

Total suite: **98 tests passing**.

## Still open

- SSO/OIDC instead of manually issued API keys.
- Postgres row-level security and a dedicated database (or schema) per tenant.
- Alembic migrations; `create_all()` is only for local/test bootstrap.
- Immutable audit storage, retention policy, and export.
- TLS termination and a secrets manager.
