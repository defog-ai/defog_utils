import concurrent
from typing import Callable, Dict

from .utils_llm import (
    LLMResponse,
    chat_anthropic,
    chat_gemini,
    chat_openai,
    chat_together,
    chat_anthropic_async,
    chat_openai_async,
    chat_together_async,
)


def map_model_to_chat_fn(model: str) -> Callable:
    """
    Returns the appropriate chat function based on the model.
    """
    if model.startswith("claude"):
        return chat_anthropic
    if model.startswith("gemini"):
        return chat_gemini
    if model.startswith("gpt") or model in ["o1", "o1-mini", "o1-preview"]:
        return chat_openai
    if (
        model.startswith("meta-llama")
        or model.startswith("mistralai")
        or model.startswith("Qwen")
    ):
        return chat_together
    raise ValueError(f"Unknown model: {model}")


def map_model_to_chat_fn_async(model: str) -> Callable:
    """
    Returns the appropriate chat function based on the model.
    """
    if model.startswith("claude"):
        return chat_anthropic_async
    if model.startswith("gemini"):
        raise ValueError("Gemini does not support async chat")
    if model.startswith("gpt") or model in ["o1", "o1-mini", "o1-preview"]:
        return chat_openai_async
    if (
        model.startswith("meta-llama")
        or model.startswith("mistralai")
        or model.startswith("Qwen")
    ):
        return chat_together_async
    raise ValueError(f"Unknown model: {model}")


async def chat_async(
    model,
    messages,
    max_completion_tokens=4096,
    temperature=0.0,
    stop=[],
    json_mode=False,
    seed=0,
) -> LLMResponse:
    """
    Returns the response from the LLM API for a single model that is passed in.
    """
    llm_function = map_model_to_chat_fn_async(model)
    return await llm_function(
        messages=messages,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        json_mode=json_mode,
        seed=seed,
    )


def chat(
    models,
    messages,
    max_completion_tokens=4096,
    temperature=0.0,
    stop=[],
    json_mode=False,
    seed=0,
) -> Dict[str, LLMResponse]:
    """
    Returns the response from the LLM API for each of the models passed in.
    Output format is a dictionary keyed by model name.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                map_model_to_chat_fn(model),
                messages,
                model,
                max_completion_tokens,
                temperature,
                stop,
                json_mode,
                seed,
            ): model
            for model in models
        }
        responses = {
            futures[future]: future.result()
            for future in concurrent.futures.as_completed(futures)
        }
    return responses
