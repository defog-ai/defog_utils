import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

LLM_COSTS_PER_TOKEN = {
    "chatgpt-4o": {"input_cost_per1k": 0.0025, "output_cost_per1k": 0.01},
    "gpt-4o": {"input_cost_per1k": 0.0025, "output_cost_per1k": 0.01},
    "gpt-4o-mini": {"input_cost_per1k": 0.00015, "output_cost_per1k": 0.0006},
    "o1": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.06},
    "o1-preview": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.06},
    "o1-mini": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.012},
    "o3-mini": {"input_cost_per1k": 0.0011, "output_cost_per1k": 0.0044},
    "gpt-4-turbo": {"input_cost_per1k": 0.01, "output_cost_per1k": 0.03},
    "gpt-3.5-turbo": {"input_cost_per1k": 0.0005, "output_cost_per1k": 0.0015},
    "claude-3-5-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-3-5-haiku": {"input_cost_per1k": 0.00025, "output_cost_per1k": 0.00125},
    "claude-3-opus": {"input_cost_per1k": 0.015, "output_cost_per1k": 0.075},
    "claude-3-sonnet": {"input_cost_per1k": 0.003, "output_cost_per1k": 0.015},
    "claude-3-haiku": {"input_cost_per1k": 0.00025, "output_cost_per1k": 0.00125},
    "gemini-1.5-pro": {"input_cost_per1k": 0.00125, "output_cost_per1k": 0.005},
    "gemini-1.5-flash": {"input_cost_per1k": 0.000075, "output_cost_per1k": 0.0003},
    "gemini-1.5-flash-8b": {"input_cost_per1k": 0.0000375, "output_cost_per1k": 0.00015,},
    "gemini-2.0-flash": {
        "input_cost_per1k": 0.00010,
        "output_cost_per1k": 0.0004,
    },
    "deepseek-chat": {"input_cost_per1k": 0.00014, "output_cost_per1k": 0.00028,},
    "deepseek-reasoner": {"input_cost_per1k": 0.00055, "output_cost_per1k": 0.00219,},
}


@dataclass
class LLMResponse:
    content: Any
    model: str
    time: float
    input_tokens: int
    output_tokens: int
    output_tokens_details: Optional[Dict[str, int]] = None
    cost_in_cents: Optional[float] = None

    def __post_init__(self):
        if self.model in LLM_COSTS_PER_TOKEN:
            model_name = self.model
        else:
            # if there is no exact match (for example, if the model name is "gpt-4o-2024-08-06")
            # then try to find the closest match
            model_name = None
            potential_model_names = []

            # first, find all the models that have a matching prefix
            for mname in LLM_COSTS_PER_TOKEN.keys():
                if mname in self.model:
                    potential_model_names.append(mname)
            
            if len(potential_model_names) > 0:
                # if there are multiple potential matches, then find the one with the longest prefix
                model_name = max(potential_model_names, key=len)
        
        if model_name:
            self.cost_in_cents = (
                self.input_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["input_cost_per1k"]
                + self.output_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["output_cost_per1k"]
                * 100
            )


def chat_anthropic(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
) -> LLMResponse:
    """
    Returns the response from the Anthropic API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Note that anthropic doesn't have explicit json mode api constraints, nor does it have a seed parameter.
    """
    from anthropic import Anthropic

    client_anthropic = Anthropic()
    t = time.time()
    if len(messages) >= 1 and messages[0].get("role") == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        sys_msg = ""
    response = client_anthropic.messages.create(
        system=sys_msg,
        messages=messages,
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        stop_sequences=stop,
    )
    if response.stop_reason == "max_tokens":
        raise Exception("Max tokens reached")
    if len(response.content) == 0:
        raise Exception("Max tokens reached")
    return LLMResponse(
        model=model,
        content=response.content[0].text,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


async def chat_anthropic_async(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
) -> LLMResponse:
    """
    Returns the response from the Anthropic API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Note that anthropic doesn't have explicit json mode api constraints, nor does it have a seed parameter.
    """
    from anthropic import AsyncAnthropic

    client_anthropic = AsyncAnthropic()
    t = time.time()
    if len(messages) >= 1 and messages[0].get("role") == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        sys_msg = ""
    response = await client_anthropic.messages.create(
        system=sys_msg,
        messages=messages,
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        stop_sequences=stop,
        timeout=timeout
    )
    if response.stop_reason == "max_tokens":
        raise Exception("Max tokens reached")
    if len(response.content) == 0:
        raise Exception("Max tokens reached")
    return LLMResponse(
        model=model,
        content=response.content[0].text,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
    )


def chat_openai(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    base_url: str = "https://api.openai.com/v1/",
    api_key: str = os.environ.get("OPENAI_API_KEY", ""),
) -> LLMResponse:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    We use max_completion_tokens here, instead of using max_tokens. This is to support o1 models.
    Note this function also supports DeepSeek models as it uses the same API. Simply use the base URL "https://api.deepseek.com"
    """
    from openai import OpenAI

    client_openai = OpenAI(
        base_url=base_url,
        api_key=api_key
    )
    t = time.time()
    if model in ["o1-mini", "o1-preview"]:
        if messages[0].get("role") == "system":
            sys_msg = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = sys_msg + messages[0]["content"]

        response = client_openai.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
        )
    else:
        if response_format:
            response = client_openai.beta.chat.completions.parse(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                response_format=response_format,
                seed=seed,
            )
        else:
            response = client_openai.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                seed=seed,
            )
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        raise Exception("Max tokens reached")

    if response_format and model not in ["o1-mini", "o1-preview"]:
        content = response.choices[0].message.parsed
    else:
        content = response.choices[0].message.content

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        output_tokens_details=response.usage.completion_tokens_details,
    )


async def chat_openai_async(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
    base_url: str = "https://api.openai.com/v1/",
    api_key: str = os.environ.get("OPENAI_API_KEY", ""),
    prediction: Dict[str,str] = None,
    reasoning_effort=None,
) -> LLMResponse:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    We use max_completion_tokens here, instead of using max_tokens. This is to support o1 models.
    Note this function also supports DeepSeek models as it uses the same API. Simply use the base URL "https://api.deepseek.com"
    """
    from openai import AsyncOpenAI

    client_openai = AsyncOpenAI(
        base_url=base_url,
        api_key=api_key
    )
    t = time.time()
    
    if model in ["o1-mini", "o1-preview", "o1", "deepseek-chat", "deepseek-reasoner", "o3-mini"]:
        # find any system message, save its value, and remove it from the list of messages
        sys_msg = None
        for i in range(len(messages)):
            if messages[i].get("role") == "system":
                sys_msg = messages.pop(i)["content"]
                break
        
        # if system message is not None, then prepend it to the first user message
        if sys_msg:
            for i in range(len(messages)):
                if messages[i].get("role") == "user":
                    messages[i]["content"] = sys_msg + "\n" + messages[i]["content"]
                    break
        
        print(messages)
    
    request_params = {
        "messages": messages,
        "model": model,
        "max_completion_tokens": max_completion_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
        "store": store,
        "metadata": metadata,
        "timeout": timeout,
        "response_format": response_format,
    }

    if model in ["gpt-4o", "gpt-4o-mini"] and prediction:
        request_params["prediction"] = prediction
        del request_params["max_completion_tokens"]
        del request_params["response_format"] # completion with prediction output does not support max_completion_tokens and response_format
    
    if model.startswith("o") or model == "deepseek-reasoner":
        del request_params["temperature"]
    
    if model in ["o1-mini", "o1-preview", "deepseek-chat", "deepseek-reasoner"]:
        del request_params["response_format"]
    
    if model.startswith("o") and reasoning_effort is not None:
        request_params["reasoning_effort"] = reasoning_effort
    
    if "response_format" in request_params and request_params["response_format"]:
        del request_params["stop"] # cannot have stop when using response_format, as that often leads to invalid JSON
        response = await client_openai.beta.chat.completions.parse(**request_params)
        content = response.choices[0].message.parsed
    else:
        response = await client_openai.chat.completions.create(**request_params)
        content = response.choices[0].message.content

    if response.choices[0].finish_reason == "length":
        print("Max tokens reached")
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        print("Empty response")
        raise Exception("No response")
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
        output_tokens_details=response.usage.completion_tokens_details,
    )


def chat_together(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
) -> LLMResponse:
    """
    Returns the response from the Together API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Together's max_tokens refers to the maximum completion tokens.
    Together doesn't have explicit json mode api constraints.
    """
    from together import Together

    client_together = Together()
    t = time.time()
    response = client_together.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
    )
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        raise Exception("Max tokens reached")
    return LLMResponse(
        model=model,
        content=response.choices[0].message.content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )


async def chat_together_async(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
) -> LLMResponse:
    """
    Returns the response from the Together API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Together's max_tokens refers to the maximum completion tokens.
    Together doesn't have explicit json mode api constraints.
    """
    from together import AsyncTogether

    client_together = AsyncTogether(timeout=timeout)
    t = time.time()
    response = await client_together.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
    )
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        raise Exception("Max tokens reached")
    return LLMResponse(
        model=model,
        content=response.choices[0].message.content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )


def chat_gemini(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    t = time.time()
    if messages[0]["role"] == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        system_msg = None

    message = "\n".join([i["content"] for i in messages])

    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_msg,
        max_output_tokens=max_completion_tokens,
        stop_sequences=stop,
    )

    if response_format:
        # use Pydantic classes for response_format
        generation_config.response_mime_type = "application/json"
        generation_config.response_schema = response_format

        del generation_config.stop_sequences

    try:
        response = client.models.generate_content(
            model=model,
            contents=message,
            config=generation_config,
        )
        content = response.text
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    if response_format:
        # convert the content into Pydantic class
        content = response_format.model_validate_json(content)

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
    )


async def chat_gemini_async(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
    timeout=100,  # does not have timeout method
    prediction=None,
    reasoning_effort=None,
) -> LLMResponse:
    from google import genai
    from google.genai import types

    client = genai.Client(
        api_key=os.getenv("GEMINI_API_KEY"),
    )
    t = time.time()
    if messages[0]["role"] == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        system_msg = None

    message = "\n".join([i["content"] for i in messages])

    generation_config = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_msg,
        max_output_tokens=max_completion_tokens,
        stop_sequences=stop,
    )

    if response_format:
        # use Pydantic classes for response_format
        generation_config.response_mime_type = "application/json"
        generation_config.response_schema = response_format

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=message,
            config=generation_config,
        )
        content = response.text
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    if response_format:
        # convert the content into Pydantic class
        content = response_format.model_validate_json(content)

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
    )
