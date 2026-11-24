from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from app.core.config import settings
from app.rag.pipeline import DocumentInput, IngestionReport, ingest_text_records

logger = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


@dataclass(frozen=True)
class S3Object:
    key: str
    size: int
    etag: str | None = None


class S3DocumentSource:
    """Minimal S3 reader for policy/filing documents."""

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        *,
        region: str | None = None,
        profile: str | None = None,
        endpoint_url: str | None = None,
        max_object_bytes: int | None = None,
        client=None,
    ):
        self.bucket = bucket
        self.prefix = prefix
        self.max_object_bytes = max_object_bytes or settings.s3_max_object_bytes
        self._client = client or self._make_client(
            region or settings.aws_region,
            profile or settings.aws_profile,
            endpoint_url if endpoint_url is not None else settings.aws_endpoint_url,
        )

    @staticmethod
    def _make_client(region, profile, endpoint_url):
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - dependency optional
            raise RuntimeError(
                "S3 ingestion requires boto3. Run: pip install -r requirements-aws.txt"
            ) from exc
        session = boto3.Session(
            profile_name=profile or None,
            region_name=region,
        )
        return session.client("s3", endpoint_url=endpoint_url or None)

    def list_documents(self) -> list[S3Object]:
        documents: list[S3Object] = []
        request = {"Bucket": self.bucket, "Prefix": self.prefix}

        while True:
            response = self._client.list_objects_v2(**request)
            for item in response.get("Contents", []):
                key = item.get("Key", "")
                if Path(key).suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                size = int(item.get("Size", 0))
                if size > self.max_object_bytes:
                    logger.warning(
                        "Skipping s3://%s/%s: %s bytes exceeds limit %s",
                        self.bucket,
                        key,
                        size,
                        self.max_object_bytes,
                    )
                    continue
                documents.append(
                    S3Object(
                        key=key,
                        size=size,
                        etag=str(item["ETag"]).strip('"') if item.get("ETag") else None,
                    )
                )

            if not response.get("IsTruncated"):
                break
            request["ContinuationToken"] = response["NextContinuationToken"]

        return documents

    def read_document(self, key: str) -> tuple[str, str | None]:
        response = self._client.get_object(Bucket=self.bucket, Key=key)
        body = response["Body"].read()
        if len(body) > self.max_object_bytes:
            raise ValueError(
                f"s3://{self.bucket}/{key} exceeds the maximum object size"
            )

        suffix = Path(key).suffix.lower()
        if suffix == ".pdf":
            text = _pdf_bytes_to_text(body)
        else:
            text = body.decode("utf-8", errors="replace")

        return text, response.get("VersionId")


def _pdf_bytes_to_text(data: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".pdf") as temp:
        temp.write(data)
        temp.flush()
        reader = PdfReader(temp.name)
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def ingest_s3_documents(
    bucket: str | None = None,
    prefix: str | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    reset: bool = False,
    tenant_id: str = "default",
    s3_client=None,
    max_object_bytes: int | None = None,
) -> IngestionReport:
    bucket = bucket or settings.s3_bucket
    if not bucket:
        raise ValueError("An S3 bucket is required (request field or S3_BUCKET).")

    source = S3DocumentSource(
        bucket=bucket,
        prefix=prefix if prefix is not None else settings.s3_prefix,
        max_object_bytes=max_object_bytes,
        client=s3_client,
    )

    records: list[DocumentInput] = []
    for obj in source.list_documents():
        text, version_id = source.read_document(obj.key)
        if not text.strip():
            logger.warning("Skipping empty document s3://%s/%s", bucket, obj.key)
            continue
        records.append(
            DocumentInput(
                source=f"s3://{bucket}/{obj.key}",
                text=text,
                metadata={
                    "source_type": "s3",
                    "s3_bucket": bucket,
                    "s3_key": obj.key,
                    "s3_etag": obj.etag,
                    "s3_version_id": version_id,
                },
            )
        )

    return ingest_text_records(
        records,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        reset=reset,
        tenant_id=tenant_id,
    )
