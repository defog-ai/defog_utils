import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any


@dataclass
class LLMResponse:
    content: Any
    time: float
    input_tokens: int
    output_tokens: int
    output_tokens_details: Optional[Dict[str, int]] = None


def chat_anthropic(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
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
        print("Max tokens reached")
        return None
    if len(response.content) == 0:
        print("Empty response")
        return None
    return LLMResponse(
        response.content[0].text,
        round(time.time() - t, 3),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


async def chat_anthropic_async(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
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
    )
    if response.stop_reason == "max_tokens":
        print("Max tokens reached")
        return None
    if len(response.content) == 0:
        print("Empty response")
        return None
    return LLMResponse(
        response.content[0].text,
        round(time.time() - t, 3),
        response.usage.input_tokens,
        response.usage.output_tokens,
    )


def chat_openai(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    We use max_completion_tokens here, instead of using max_tokens. This is to support o1 models.
    """
    from openai import OpenAI

    client_openai = OpenAI()
    t = time.time()
    if model in ["o1-mini", "o1-preview", "o1"]:
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
        if response_format or json_mode:
            response = client_openai.beta.chat.completions.parse(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                response_format={"type": "json_object"} if json_mode else response_format,
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
        print("Max tokens reached")
        return None
    if len(response.choices) == 0:
        print("Empty response")
        return None
    
    if response_format and model not in ["o1-mini", "o1-preview", "o1"]:
        content = response.choices[0].message.parsed
    else:
        content = response.choices[0].message.content
    
    return LLMResponse(
        content,
        round(time.time() - t, 3),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.completion_tokens_details,
    )


async def chat_openai_async(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    We use max_completion_tokens here, instead of using max_tokens. This is to support o1 models.
    """
    from openai import AsyncOpenAI

    client_openai = AsyncOpenAI()
    t = time.time()
    if model in ["o1-mini", "o1-preview", "o1"]:
        if messages[0].get("role") == "system":
            sys_msg = messages[0]["content"]
            messages = messages[1:]
            messages[0]["content"] = sys_msg + messages[0]["content"]
        
        response = await client_openai.chat.completions.create(
            messages=messages,
            model=model,
            max_completion_tokens=max_completion_tokens,
        )
    else:
        if response_format or json_mode:
            response = await client_openai.beta.chat.completions.parse(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                response_format={"type": "json_object"} if json_mode else response_format,
                seed=seed,
            )
        else:
            response = await client_openai.chat.completions.create(
                messages=messages,
                model=model,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
                stop=stop,
                seed=seed,
            )
    
    if response_format and model not in ["o1-mini", "o1-preview", "o1"]:
        content = response.choices[0].message.parsed
    else:
        content = response.choices[0].message.content
    
    if response.choices[0].finish_reason == "length":
        print("Max tokens reached")
        return None
    if len(response.choices) == 0:
        print("Empty response")
        return None
    return LLMResponse(
        content,
        round(time.time() - t, 3),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
        response.usage.completion_tokens_details,
    )


def chat_together(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
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
        print("Max tokens reached")
        return None
    if len(response.choices) == 0:
        print("Empty response")
        return None
    return LLMResponse(
        response.choices[0].message.content,
        round(time.time() - t, 3),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


async def chat_together_async(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
    """
    Returns the response from the Together API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Together's max_tokens refers to the maximum completion tokens.
    Together doesn't have explicit json mode api constraints.
    """
    from together import AsyncTogether

    client_together = AsyncTogether()
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
        print("Max tokens reached")
        return None
    if len(response.choices) == 0:
        print("Empty response")
        return None
    return LLMResponse(
        response.choices[0].message.content,
        round(time.time() - t, 3),
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def chat_gemini(
    messages: List[Dict[str, str]],
    model: str = "gemini-1.5-pro",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    response_format=None,
    seed: int = 0,
) -> Optional[LLMResponse]:
    import google.generativeai as genai

    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    t = time.time()
    generation_config = {
        "temperature": temperature,
        "max_output_tokens": max_completion_tokens,
        "response_mime_type": "application/json" if json_mode else "text/plain",
        "stop_sequences": stop,
        # "seed": seed, # seed is not supported in the current version
    }
    if messages[0]["role"] == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        system_msg = None
    final_msg = messages[-1]["content"]
    messages = messages[:-1]
    for msg in messages:
        if msg["role"] != "user":
            msg["role"] = "model"
    client_gemini = genai.GenerativeModel(
        model, generation_config=generation_config, system_instruction=system_msg
    )
    chat = client_gemini.start_chat(
        history=messages,
    )
    response = chat.send_message(final_msg)
    if len(response.candidates) == 0:
        print("Empty response")
        return None
    if (
        response.candidates[0].finish_reason.value != 1
    ):  # 1 is the finish reason for STOP
        print("Max tokens reached")
        return None
    return LLMResponse(
        content=response.candidates[0].content.parts[0].text,
        time=round(time.time() - t, 3),
        input_tokens=response.usage_metadata.prompt_token_count,
        output_tokens=response.usage_metadata.candidates_token_count,
    )
