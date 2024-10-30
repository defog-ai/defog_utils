import concurrent
from typing import Callable, Dict

from .utils_llm import LLMResponse, chat_anthropic, chat_gemini, chat_openai, chat_together

def map_model_to_chat_fn(model: str) -> Callable:
    """
    Returns the appropriate chat function based on the model.
    """
    if model.startswith("claude"):
        return chat_anthropic
    if model.startswith("gemini"):
        return chat_gemini
    if model.startswith("gpt"):
        return chat_openai
    if model.startswith("meta-llama") or model.startswith("mistralai") or model.startswith("Qwen"):
        return chat_together
    raise ValueError(f"Unknown model: {model}")

def chat(models, messages, max_tokens=8192, temperature=0.0, stop=[], json_mode=False, seed=0) -> Dict[str, LLMResponse]:
    """
    Returns the response from the LLM API for each of the models passed in.
    Output format is a dictionary keyed by model name.
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {executor.submit(map_model_to_chat_fn(model), messages, model, max_tokens, temperature, stop, json_mode, seed): model for model in models}
        responses = {futures[future]: future.result() for future in concurrent.futures.as_completed(futures)}
    return responses