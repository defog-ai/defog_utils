from dataclasses import dataclass
import time
from typing import Optional, List, Dict
from anthropic import Anthropic
from openai import OpenAI
from together import Together

client_anthropic = Anthropic()
client_openai = OpenAI()
client_together = Together()


@dataclass
class LLMResponse:
    content: str
    time: float
    input_tokens: int
    output_tokens: int


def chat_anthropic(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    seed: int = 0,
) -> Optional[LLMResponse]:
    """
    Returns the response from the Anthropic API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Note that anthropic doesn't have explicit json mode api constraints, nor does it have a seed parameter.
    """
    t = time.time()
    if len(messages) >= 1 and messages[0].get("role") == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        sys_msg = None
    response = client_anthropic.messages.create(
        system=sys_msg,
        messages=messages,
        model=model,
        max_tokens=max_tokens,
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
    max_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    seed: int = 0,
) -> Optional[LLMResponse]:
    """
    Returns the response from the OpenAI API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    """
    t = time.time()
    response = client_openai.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        stop=stop,
        response_format={"type": "json_object"} if json_mode else None,
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


def chat_together(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    max_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    json_mode: bool = False,
    seed: int = 0,
) -> Optional[LLMResponse]:
    """
    Returns the response from the Together API, the time taken to generate the response, the number of input tokens used, and the number of output tokens used.
    Together's max_tokens refers to the maximum completion tokens, not the maximum total tokens, hence requires calculating 8192 - input_tokens.
    Together doesn't have explicit json mode api constraints.
    """
    t = time.time()
    response = client_together.chat.completions.create(
        messages=messages,
        model=model,
        max_tokens=max_tokens,
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
