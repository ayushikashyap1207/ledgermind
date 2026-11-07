import pytest

from app.core.llm import LLMClient, RetryingLLMClient


class FlakyClient(LLMClient):
    def __init__(self, failures: int, error: Exception | None = None):
        self.failures = failures
        self.error = error or RuntimeError("rate limit exceeded")
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise self.error
        return "ok"


def test_transient_error_is_retried():
    inner = FlakyClient(failures=1)
    client = RetryingLLMClient(inner, max_retries=1, backoff_seconds=0)

    assert client.complete(system="s", user="u") == "ok"
    assert inner.calls == 2


def test_non_retryable_error_is_not_retried():
    inner = FlakyClient(
        failures=1, error=RuntimeError("401 unauthorized: invalid api key")
    )
    client = RetryingLLMClient(inner, max_retries=3, backoff_seconds=0)

    with pytest.raises(RuntimeError):
        client.complete(system="s", user="u")
    assert inner.calls == 1


def test_retries_are_bounded():
    inner = FlakyClient(failures=5)
    client = RetryingLLMClient(inner, max_retries=2, backoff_seconds=0)

    with pytest.raises(RuntimeError):
        client.complete(system="s", user="u")
    assert inner.calls == 3
