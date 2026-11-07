from app.core.config import LLMProvider, Settings
from app.core.llm import OllamaClient, get_llm


def test_ollama_provider_constructs_a_local_client_without_an_api_key():
    cfg = Settings(
        llm_provider=LLMProvider.OLLAMA,
        llm_model="llama3.2:3b",
        ollama_base_url="http://127.0.0.1:11434/v1",
        llm_max_retries=0,
    )

    assert cfg.active_api_key() == "ollama"
    assert isinstance(get_llm(cfg), OllamaClient)
