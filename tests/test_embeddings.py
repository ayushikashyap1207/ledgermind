import numpy as np

from app.rag.embeddings import (
    HashEmbeddingFunction,
    SentenceTransformerEmbeddingFunction,
    embedding_profile_for_model,
)


class FakeSentenceTransformer:
    def __init__(self):
        self.texts: list[str] = []

    def encode(
        self,
        texts,
        batch_size=None,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        self.texts = list(texts)
        return np.array([[1.0, 0.0] for _ in texts])


def test_bge_profile_uses_query_instruction_only():
    profile = embedding_profile_for_model("BAAI/bge-small-en-v1.5")

    assert "Represent this sentence" in profile.query_prefix
    assert profile.document_prefix == ""


def test_e5_profile_uses_both_prefixes():
    profile = embedding_profile_for_model("intfloat/e5-small-v2")

    assert profile.query_prefix == "query: "
    assert profile.document_prefix == "passage: "


def test_nomic_profile_uses_both_prefixes():
    profile = embedding_profile_for_model("nomic-ai/nomic-embed-text-v1.5")

    assert profile.query_prefix == "search_query: "
    assert profile.document_prefix == "search_document: "


def test_hash_embedding_name_and_query_method():
    embedding = HashEmbeddingFunction()

    assert embedding.name() == "hash"
    assert embedding.collection_name_suffix() == "hash-384"
    assert embedding.embed_query("hello") == embedding.embed_text("hello")


def test_sentence_transformer_applies_document_and_query_prefixes():
    fake = FakeSentenceTransformer()
    embedding = SentenceTransformerEmbeddingFunction(
        "intfloat/e5-small-v2", model=fake
    )

    vectors = embedding.embed_documents(["invoice terms"])
    assert len(vectors) == 1
    assert fake.texts == ["passage: invoice terms"]

    query_vector = embedding.embed_query("when is the invoice due?")
    assert query_vector == [1.0, 0.0]
    assert fake.texts == ["query: when is the invoice due?"]


def test_sentence_transformer_name_includes_model():
    embedding = SentenceTransformerEmbeddingFunction(
        "BAAI/bge-small-en-v1.5", model=FakeSentenceTransformer()
    )

    assert embedding.name() == "sentence_transformer"
    assert embedding.collection_name_suffix() == "st-BAAI-bge-small-en-v1.5"
