import pytest

from app.core.config import LLMProvider, Settings
from app.core.llm import BedrockClient, GuardrailViolation
from app.rag.pipeline import RetrievedChunk
from app.router.synthesis import synthesize_answer


class FakeBedrockClient:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def converse(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def make_config(guardrail: bool = False) -> Settings:
    return Settings(
        llm_provider=LLMProvider.BEDROCK,
        llm_model="test-bedrock-model",
        aws_region="us-east-1",
        bedrock_guardrail_enabled=guardrail,
        bedrock_guardrail_id="guardrail-123" if guardrail else None,
        bedrock_guardrail_version="1",
        llm_max_retries=0,
    )


def test_bedrock_converse_returns_text():
    fake = FakeBedrockClient(
        {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": "grounded answer"}]}},
        }
    )
    client = BedrockClient(make_config(), client=fake)

    result = client.complete(system="system prompt", user="user prompt")

    assert result == "grounded answer"
    assert fake.kwargs["modelId"] == "test-bedrock-model"
    assert fake.kwargs["system"] == [{"text": "system prompt"}]
    assert fake.kwargs["messages"][0]["content"] == [{"text": "user prompt"}]


def test_bedrock_guardrail_config_is_forwarded():
    fake = FakeBedrockClient(
        {
            "stopReason": "end_turn",
            "output": {"message": {"content": [{"text": "ok"}]}},
        }
    )
    client = BedrockClient(make_config(guardrail=True), client=fake)

    assert client.complete("system", "user") == "ok"
    assert fake.kwargs["guardrailConfig"] == {
        "guardrailIdentifier": "guardrail-123",
        "guardrailVersion": "1",
    }


def test_bedrock_guardrail_intervention_raises():
    fake = FakeBedrockClient(
        {"stopReason": "guardrail_intervened", "output": {"message": {"content": []}}}
    )
    client = BedrockClient(make_config(guardrail=True), client=fake)

    with pytest.raises(GuardrailViolation):
        client.complete("system", "user")


class GuardrailLLM:
    def complete(self, system: str, user: str) -> str:
        raise GuardrailViolation("blocked")


def test_synthesis_returns_a_safe_refusal_on_guardrail_block():
    chunk = RetrievedChunk(
        chunk_id="doc::1",
        source="s3://bucket/policy.txt",
        chunk_index=0,
        text="The procurement threshold is 25,000 USD.",
        score=1.0,
    )

    result = synthesize_answer(
        "What is the procurement threshold?",
        "needs_rag",
        None,
        [chunk],
        llm=GuardrailLLM(),
    )

    assert result.method == "guardrail"
    assert result.insufficient_evidence
    assert "safety policy" in result.text
    assert result.citations == []
