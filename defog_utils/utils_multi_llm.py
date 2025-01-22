import concurrent
import asyncio
from typing import Callable, Dict
import os

from .utils_llm import (
    LLMResponse,
    chat_anthropic,
    chat_gemini,
    chat_openai,
    chat_together,
    chat_anthropic_async,
    chat_gemini_async,
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
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("chatgpt"):
        return chat_openai
    if model.startswith("deepseek"):
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
        return chat_gemini_async
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("chatgpt"):
        return chat_openai_async
    if model.startswith("deepseek"):
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
    response_format=None,
    seed=0,
    store=True,
    metadata=None,
    timeout=100, # in seconds
    backup_model=None,
    prediction=None
) -> LLMResponse:
    """
    Returns the response from the LLM API for a single model that is passed in.
    Includes retry logic with exponential backoff for up to 3 attempts.
    """
    llm_function = map_model_to_chat_fn_async(model)
    max_retries = 3
    base_delay = 1  # Initial delay in seconds

    latest_error = None

    for attempt in range(max_retries):
        try:
            if attempt > 0 and backup_model is not None:
                # For the first attempt, use the original model
                # For subsequent attempts, use the backup model if it is provided
                model = backup_model
                llm_function = map_model_to_chat_fn_async(model)
            if not model.startswith("deepseek"):
                if prediction and "gpt-4o" in model:
                    # predicted output completion does not support response_format and max_completion_tokens
                    return await llm_function(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stop=stop,
                        seed=seed,
                        store=store,
                        metadata=metadata,
                        timeout=timeout,
                        prediction=prediction
                    )
                else:
                    return await llm_function(
                        model=model,
                        messages=messages,
                        max_completion_tokens=max_completion_tokens,
                        temperature=temperature,
                        stop=stop,
                        response_format=response_format,
                        seed=seed,
                        store=store,
                        metadata=metadata,
                        timeout=timeout,
                    )
            else:
                if not os.getenv("DEEPSEEK_API_KEY"):
                    raise Exception("DEEPSEEK_API_KEY is not set")
                return await llm_function(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    stop=stop,
                    response_format=response_format,
                    seed=seed,
                    store=store,
                    metadata=metadata,
                    timeout=timeout,
                    base_url="https://api.deepseek.com",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                )
        except Exception as e:
            delay = base_delay * (2 ** attempt)  # Exponential backoff
            print(f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...", flush=True)
            print(f"Error: {e}", flush=True)
            latest_error = e
            await asyncio.sleep(delay)
    
    # If we get here, all attempts failed
    raise Exception("All attempts at calling the chat_async function failed. The latest error was: ", latest_error)


def chat(
    models,
    messages,
    max_completion_tokens=4096,
    temperature=0.0,
    stop=[],
    response_format=None,
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
                response_format,
                seed,
            ): model
            for model in models
        }
        responses = {
            futures[future]: future.result()
            for future in concurrent.futures.as_completed(futures)
        }
    return responses
