from app.rag.bm25 import BM25Index, tokenize
from app.rag.hybrid import (
    cosine_similarity,
    mmr_select,
    normalize_scores,
    reciprocal_rank_fusion,
)

DOCS = {
    "expense": (
        "Approved expense reports are reimbursed within 10 business days. "
        "Managers approve submitted expense reports within 3 business days."
    ),
    "procurement": (
        "Purchases above 50,000 USD require VP Finance approval. "
        "The procurement approval threshold is 25,000 USD."
    ),
    "vendor": (
        "Invoices are due Net 45. Invoice disputes must be raised within "
        "20 calendar days of invoice receipt."
    ),
}


def test_tokenize_normalizes_case_and_punctuation():
    assert tokenize("Net-45, FY2026!") == ["net", "45", "fy2026"]


def test_tokenize_removes_function_words_but_keeps_policy_terms():
    assert tokenize("What is the total within 45 days?") == ["total", "within", "45", "days"]


def test_bm25_ranks_exact_reimbursement_document_first():
    index = BM25Index(DOCS)

    hits = index.search("What is the reimbursement timeline for approved expenses?")

    assert hits
    assert hits[0].doc_id == "expense"


def test_bm25_ranks_dispute_document_first():
    index = BM25Index(DOCS)

    hits = index.search("What happens if invoice disputes are not raised in time?")

    assert hits
    assert hits[0].doc_id == "vendor"


def test_empty_bm25_index_is_safe():
    index = BM25Index({})

    assert len(index) == 0
    assert index.search("anything") == []
    assert index.score_all("anything") == {}


def test_normalize_scores_handles_zero_and_equal_values():
    assert normalize_scores({}) == {}
    assert normalize_scores({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}
    assert normalize_scores({"a": 2.0, "b": 2.0}) == {"a": 1.0, "b": 1.0}
    assert normalize_scores({"a": 1.0, "b": 3.0}) == {"a": 0.0, "b": 1.0}


def test_rrf_rewards_documents_present_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]])

    assert fused["b"] > fused["a"]
    assert fused["b"] > fused["c"]
    assert fused["d"] > 0.0


def test_mmr_prefers_relevance_then_diversity():
    selected = mmr_select(
        candidate_ids=["a", "b", "c"],
        relevance={"a": 1.0, "b": 0.99, "c": 0.5},
        embeddings={"a": [1.0, 0.0], "b": [1.0, 0.0], "c": [0.0, 1.0]},
        top_k=3,
        lambda_=0.5,
    )

    assert selected == ["a", "c", "b"]


def test_cosine_similarity_of_normalized_vectors():
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0
