import pytest

from app.rag.s3 import S3DocumentSource, ingest_s3_documents
from app.rag.store import get_collection, reset_collection


class FakeBody:
    def __init__(self, data: bytes):
        self.data = data

    def read(self) -> bytes:
        return self.data


class FakeS3Client:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.get_calls: list[str] = []

    def list_objects_v2(self, **kwargs):
        prefix = kwargs.get("Prefix", "")
        contents = [
            {"Key": key, "Size": len(value), "ETag": '"etag-1"'}
            for key, value in self.objects.items()
            if key.startswith(prefix)
        ]
        return {"Contents": contents, "IsTruncated": False}

    def get_object(self, **kwargs):
        key = kwargs["Key"]
        self.get_calls.append(key)
        return {"Body": FakeBody(self.objects[key]), "VersionId": "version-1"}


def test_s3_source_filters_unsupported_files():
    source = S3DocumentSource(
        bucket="test-bucket",
        prefix="docs/",
        client=FakeS3Client(
            {
                "docs/policy.txt": b"policy text",
                "docs/ignore.exe": b"binary",
                "other/notes.md": b"other",
            }
        ),
    )

    documents = source.list_documents()

    assert [document.key for document in documents] == ["docs/policy.txt"]


def test_s3_ingestion_creates_tenant_scoped_chunks():
    tenant_id = "s3-test-tenant"
    reset_collection(tenant_id)
    try:
        fake = FakeS3Client(
            {
                "docs/policy.txt": b"The procurement approval threshold is 25,000 USD.",
                "docs/msa.md": b"Invoices are due Net 45 from the issue date.",
            }
        )

        report = ingest_s3_documents(
            bucket="test-bucket",
            prefix="docs/",
            tenant_id=tenant_id,
            reset=True,
            s3_client=fake,
        )

        assert report.files == 2
        assert report.chunks >= 2
        assert set(fake.get_calls) == {"docs/policy.txt", "docs/msa.md"}

        stored = get_collection(tenant_id).get(include=["metadatas"])
        metadata = stored["metadatas"]
        sources = {item["source"] for item in metadata}
        assert "s3://test-bucket/docs/policy.txt" in sources
        assert all(item["tenant_id"] == tenant_id for item in metadata)
        assert all(item["source_type"] == "s3" for item in metadata)
        assert all(item["s3_version_id"] == "version-1" for item in metadata)
    finally:
        reset_collection(tenant_id)


def test_s3_ingestion_requires_a_bucket(monkeypatch):
    from app.core.config import settings

    monkeypatch.setattr(settings, "s3_bucket", None)

    with pytest.raises(ValueError):
        ingest_s3_documents(bucket=None, s3_client=FakeS3Client({}))
