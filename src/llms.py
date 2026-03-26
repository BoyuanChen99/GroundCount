# src/llms.py
from typing import Optional

from all_llms.base import LLM
# Local hf LLMs
from all_llms.qwen3coder import Qwen3Coder
from all_llms.qwen3 import Qwen3


def init_llm(
        model_name: str,
        quantization: Optional[str] = None,  # Only relevant to local hf models
        use_openrouter: bool = False,  # Only relevant to online APIs
        api_key: Optional[str] = None
    ) -> LLM:
    if "qwen3-coder" in model_name.lower() and "Qwen" in model_name:
        return Qwen3Coder(model_name=model_name, quantization=quantization)
    elif "qwen3" in model_name.lower() and "Qwen" in model_name:
        return Qwen3(model_name=model_name, quantization=quantization)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")