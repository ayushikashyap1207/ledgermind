#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/usr/local/bin/python3}

$PYTHON_BIN -m pip install -r requirements.txt
$PYTHON_BIN -m ingestion.seed_synthetic_data
$PYTHON_BIN -m ingestion.ingest_documents docs/rag_corpus --chunk-size 700 --chunk-overlap 120 --reset

exec $PYTHON_BIN -m uvicorn app.main:app --host 0.0.0.0 --port 8000
