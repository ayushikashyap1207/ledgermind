-- Runs automatically on first postgres container start (mounted into
-- /docker-entrypoint-initdb.d/). Creates the read-only role that
-- app/sql/executor.py connects as for ALL generated-query execution —
-- this is the "run the DB connection itself as a read-only role at the
-- Postgres level" defense-in-depth requirement from the spec.

CREATE ROLE ledgermind_ro WITH LOGIN PASSWORD 'ledgermind_ro';

-- Wait for the main app role/schema to exist before granting; this
-- script runs after the main DB/user are created by the standard
-- POSTGRES_USER/POSTGRES_DB env vars in docker-compose.yml.
GRANT CONNECT ON DATABASE ledgermind TO ledgermind_ro;

\c ledgermind

GRANT USAGE ON SCHEMA public TO ledgermind_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO ledgermind_ro;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO ledgermind_ro;

-- Explicitly revoke write privileges in case of future default changes.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public FROM ledgermind_ro;
