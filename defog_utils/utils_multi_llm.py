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
import traceback


def map_model_to_chat_fn(model: str) -> Callable:
    """
    Returns the appropriate chat function based on the model.
    """
    if model.startswith("claude"):
        return chat_anthropic
    if model.startswith("gemini"):
        return chat_gemini
    if (
        model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("chatgpt")
        or model.startswith("o3")
    ):
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
    if (
        model.startswith("gpt")
        or model.startswith("o1")
        or model.startswith("chatgpt")
        or model.startswith("o3")
    ):
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
    timeout=100,  # in seconds
    backup_model=None,
    prediction=None,
    reasoning_effort=None,
    tools=None,
    tool_choice=None,
    max_retries=3,
) -> LLMResponse:
    """
    Returns the response from the LLM API for a single model that is passed in.
    Includes retry logic with exponential backoff for up to 3 attempts.
    """
    llm_function = map_model_to_chat_fn_async(model)
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
                return await llm_function(
                    model=model,
                    messages=messages,
                    max_completion_tokens=max_completion_tokens,
                    temperature=temperature,
                    stop=stop,
                    response_format=response_format,
                    seed=seed,
                    tools=tools,
                    tool_choice=tool_choice,
                    store=store,
                    metadata=metadata,
                    timeout=timeout,
                    prediction=prediction,
                    reasoning_effort=reasoning_effort,
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
                    prediction=prediction,
                    reasoning_effort=reasoning_effort,
                    base_url="https://api.deepseek.com",
                    api_key=os.getenv("DEEPSEEK_API_KEY"),
                )
        except Exception as e:
            delay = base_delay * (2**attempt)  # Exponential backoff
            print(
                f"Attempt {attempt + 1} failed. Retrying in {delay} seconds...",
                flush=True,
            )
            print(f"Error: {e}", flush=True)
            latest_error = e
            error_trace = traceback.format_exc()
            await asyncio.sleep(delay)

    # If we get here, all attempts failed
    raise Exception(
        "All attempts at calling the chat_async function failed. The latest error traceback was: ",
        error_trace,
    )


def chat(
    models,
    messages,
    max_completion_tokens=4096,
    temperature=0.0,
    stop=[],
    response_format=None,
    seed=0,
    tools=None,
    tool_choice=None,
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
                tools,
                tool_choice,
            ): model
            for model in models
        }
        responses = {
            futures[future]: future.result()
            for future in concurrent.futures.as_completed(futures)
        }
    return responses
