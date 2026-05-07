from __future__ import annotations

from pydantic_ai.exceptions import UserError
from pydantic_ai.models import infer_model

from utils.logging import get_logger

log = get_logger(__name__)


def _normalize_spec(spec: str) -> str:
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("Model spec is empty.")
    if ":" not in spec:
        raise ValueError(f"Invalid model spec '{spec}'. Expected '<provider>:<model>'.")
    provider, model = spec.split(":", 1)
    provider = provider.strip()
    model = model.strip()
    if not provider or not model:
        raise ValueError(f"Invalid model spec '{spec}'. Expected '<provider>:<model>'.")
    return f"{provider}:{model}"


def build_model(spec: str, *, strict: bool = True):
    """
    Build a PydanticAI model from an explicit format, e.g.:
      "anthropic:claude-3-5-haiku-latest"
      "ollama:llama3.1:8b"   -> local Ollama at OLLAMA_BASE_URL (default http://localhost:11434/v1)
    """
    spec = _normalize_spec(spec)

    # Local-LLM shortcut: "ollama:<model>" routes through PydanticAI's Ollama provider
    # (cost-saving alternative to anthropic). Skips inference of provider class.
    if spec.startswith("ollama:"):
        model_name = spec.split(":", 1)[1]
        try:
            import os
            from pydantic_ai.providers.ollama import OllamaProvider
            from pydantic_ai.models.openai import OpenAIChatModel
            base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
            provider = OllamaProvider(base_url=base_url)
            model = OpenAIChatModel(model_name, provider=provider)
            log.info("Built Ollama model: %s @ %s", model_name, base_url)
            return model
        except Exception as e:
            msg = f"Failed to build Ollama model '{spec}': {e}"
            if strict:
                raise ValueError(msg) from e
            log.warning(msg)
            return None

    try:
        model = infer_model(spec)
        log.info("Built model from spec: %s", spec)
        return model
    except (UserError, ValueError, ImportError) as e:
        msg = f"Failed to build model '{spec}': {e}"
        if strict:
            raise ValueError(msg) from e
        log.warning(msg)
        return None
