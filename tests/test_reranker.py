import numpy as np
import pytest

from app.rag.reranker import CrossEncoderReranker, combine_rerank_scores


class FakeCrossEncoder:
    def __init__(self):
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs, batch_size=None, show_progress_bar=False):
        self.pairs = list(pairs)
        return np.array(
            [1.0 if "expense" in document.lower() else 0.0 for _, document in pairs]
        )


def test_cross_encoder_scores_documents():
    model = FakeCrossEncoder()
    reranker = CrossEncoderReranker(model=model)

    scores = reranker.score(
        "What is the reimbursement timeline?",
        ["Expense policy: reimbursed in 10 days.", "Vendor MSA payment terms."],
    )

    assert scores == [1.0, 0.0]
    assert model.pairs[0][0] == "What is the reimbursement timeline?"


def test_cross_encoder_empty_documents_is_safe():
    reranker = CrossEncoderReranker(model=FakeCrossEncoder())

    assert reranker.score("question", []) == []


def test_combine_rerank_scores_blends_cross_and_base_scores():
    combined = combine_rerank_scores(
        base_scores={"a": 1.0, "b": 0.5},
        cross_scores={"a": 0.1, "b": 0.9},
        weight=0.7,
    )

    assert combined["a"] == pytest.approx(0.3)
    assert combined["b"] == pytest.approx(0.85)


def test_combine_rerank_scores_without_cross_scores_keeps_base():
    base = {"a": 1.0, "b": 0.5}

    assert combine_rerank_scores(base, {}) == base
