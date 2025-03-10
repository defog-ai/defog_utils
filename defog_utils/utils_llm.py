import os
import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union, Callable

from .models import (
    OpenAIToolChoice,
    OpenAIForcedFunction,
    AnthropicToolChoice,
    AnthropicForcedFunction,
)
from .utils_function_calling import get_function_specs, execute_tool, execute_tool_async
import re
import inspect
import asyncio

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
    "gemini-1.5-flash-8b": {
        "input_cost_per1k": 0.0000375,
        "output_cost_per1k": 0.00015,
    },
    "gemini-2.0-flash": {
        "input_cost_per1k": 0.00010,
        "output_cost_per1k": 0.0004,
    },
    "deepseek-chat": {
        "input_cost_per1k": 0.00014,
        "output_cost_per1k": 0.00028,
    },
    "deepseek-reasoner": {
        "input_cost_per1k": 0.00055,
        "output_cost_per1k": 0.00219,
    },
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
    tools_used: Optional[List[str]] = None

    def __post_init__(self):
        if self.model in LLM_COSTS_PER_TOKEN:
            model_name = self.model
        else:
            # Attempt partial matches if no exact match
            model_name = None
            potential_model_names = []
            for mname in LLM_COSTS_PER_TOKEN.keys():
                if mname in self.model:
                    potential_model_names.append(mname)
            if len(potential_model_names) > 0:
                model_name = max(potential_model_names, key=len)

        if model_name:
            self.cost_in_cents = (
                self.input_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["input_cost_per1k"]
                + self.output_tokens
                / 1000
                * LLM_COSTS_PER_TOKEN[model_name]["output_cost_per1k"]
            ) * 100


#
# --------------------------------------------------------------------------------
# 1) ANTHROPIC
# --------------------------------------------------------------------------------
#


def _build_anthropic_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    tools: List[Callable] = None,
    tool_choice: Union[AnthropicToolChoice, AnthropicForcedFunction] = None,
    timeout=100,
):
    """Create the parameter dict for Anthropic's .messages.create()."""
    if len(messages) >= 1 and messages[0].get("role") == "system":
        sys_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        sys_msg = ""

    params = {
        "system": sys_msg,
        "messages": messages,
        "model": model,
        "max_tokens": max_completion_tokens,
        "temperature": temperature,
        "stop_sequences": stop,
        "timeout": timeout,
    }
    if tools:
        function_specs = get_function_specs(tools, model)
        params["tools"] = function_specs
    if tool_choice:
        if tool_choice == AnthropicToolChoice.REQUIRED:
            tool_choice = "any"
        params["tool_choice"] = tool_choice

    return params, messages  # returning updated messages in case we want them


async def _process_anthropic_response(
    client,
    response,
    request_params,
    tools,
    tool_dict,
    is_async,
):
    """
    Extract content (including any tool calls) and usage info from Anthropic response.
    Handles chaining of tool calls.
    """
    from anthropic.types import ToolUseBlock

    if response.stop_reason == "max_tokens":
        raise Exception("Max tokens reached")
    if len(response.content) == 0:
        raise Exception("Max tokens reached")

    # If we have tools, handle dynamic chaining:
    tools_used = []
    if tools and len(tools) > 0:
        while True:
            # Check if the response contains a tool call
            tool_call_block = next(
                (
                    block
                    for block in response.content
                    if isinstance(block, ToolUseBlock)
                ),
                None,
            )
            if tool_call_block:
                try:
                    func_name = tool_call_block.name
                    args = tool_call_block.input
                    tool_id = tool_call_block.id
                except Exception as e:
                    raise Exception(f"Error parsing tool call: {e}")

                try:
                    tool_to_call = tool_dict[func_name]
                except KeyError:
                    raise Exception(f"Tool `{func_name}` not found.")

                # Execute tool depending on whether it is async
                try:
                    if inspect.iscoroutinefunction(tool_to_call):
                        result = await execute_tool_async(tool_to_call, args)
                    else:
                        result = execute_tool(tool_to_call, args)
                except Exception as e:
                    raise Exception(f"Error executing tool `{func_name}`: {e}")
                tools_used.append(func_name)

                # Append the tool call as an assistant response
                request_params["messages"].append(
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": func_name,
                                "input": args,
                            }
                        ],
                    }
                )

                # Append the tool result as a user response
                request_params["messages"].append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_id,
                                "content": str(result),
                            }
                        ],
                    }
                )

                # Make next call
                if is_async:
                    response = await client.messages.create(**request_params)
                else:
                    response = client.messages.create(**request_params)
            else:
                content = response.content[0].text
                break
    else:
        # No tools provided
        content = response.content[0].text

    usage = response.usage
    return content, tools_used, usage.input_tokens, usage.output_tokens


def _process_anthropic_response_handler(
    client,
    response,
    request_params: Dict[str, Any],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    is_async=False,
):
    """
    Processes Anthropic's response by determining whether to execute the response handling
    synchronously or asynchronously. This function acts as a wrapper around _process_anthropic_response,
    deciding the execution mode based on the is_async parameter.

    Parameters:
    - client: The client instance used for communication.
    - response: The response object from Anthropic.
    - request_params: A dictionary of request parameters that resulted in the response.
    - tools: A list of callable tools available for function calling.
    - tool_dict: A dictionary mapping tool names to their callable functions.
    - is_async: A boolean flag indicating whether to execute asynchronously.

    Returns:
    - The processed response content, input tokens, and output tokens from the response.
    """
    try:
        if is_async:
            return _process_anthropic_response(
                client=client,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                is_async=is_async,
            )  # Caller must await this
        else:
            return asyncio.run(
                _process_anthropic_response(
                    client=client,
                    response=response,
                    request_params=request_params,
                    tools=tools,
                    tool_dict=tool_dict,
                    is_async=is_async,
                )
            )
    except Exception as e:
        raise Exception("Error processing Anthropic response:", e)


def chat_anthropic(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: Union[AnthropicToolChoice, AnthropicForcedFunction] = None,
):
    """Synchronous Anthropic chat."""
    from anthropic import Anthropic

    t = time.time()
    client = Anthropic()
    request_params, _ = _build_anthropic_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

    response = client.messages.create(**request_params)
    content, tools_used, input_toks, output_toks = _process_anthropic_response_handler(
        client=client,
        response=response,
        request_params=request_params,
        tools=tools,
        tool_dict=tool_dict,
        is_async=False,
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
        tools_used=tools_used,
    )


async def chat_anthropic_async(
    messages: List[Dict[str, str]],
    model: str = "claude-3-5-sonnet-20241022",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: Union[AnthropicToolChoice, AnthropicForcedFunction] = None,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
):
    """Asynchronous Anthropic chat."""
    from anthropic import AsyncAnthropic

    t = time.time()
    client = AsyncAnthropic()
    params, _ = _build_anthropic_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        tools=tools,
        tool_choice=tool_choice,
    )

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in params:
        tool_dict = {tool.__name__: tool for tool in tools}

    response = await client.messages.create(**params)
    content, tools_used, input_toks, output_toks = (
        await _process_anthropic_response_handler(
            client=client,
            response=response,
            request_params=params,
            tools=tools,
            tool_dict=tool_dict,
            is_async=True,
        )
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
        tools_used=tools_used,
    )


#
# --------------------------------------------------------------------------------
# 2) OPENAI
# --------------------------------------------------------------------------------
#


def _build_openai_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: Union[OpenAIToolChoice, OpenAIForcedFunction] = None,
    prediction=None,
    reasoning_effort=None,
    store=True,
    metadata=None,
    timeout=100,
):
    """
    Build the parameter dictionary for OpenAI's chat.completions.create().
    Also handles special logic for o1-mini, o1-preview, deepseek-chat, etc.
    """
    # Potentially move system message to user message for certain model families:
    if model in [
        "o1-mini",
        "o1-preview",
        "o1",
        "deepseek-chat",
        "deepseek-reasoner",
        "o3-mini",
    ]:
        sys_msg = None
        for i in range(len(messages)):
            if messages[i].get("role") == "system":
                sys_msg = messages.pop(i)["content"]
                break
        if sys_msg:
            for i in range(len(messages)):
                if messages[i].get("role") == "user":
                    messages[i]["content"] = sys_msg + "\n" + messages[i]["content"]
                    break

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
    }

    # Tools are only supported for certain models
    if (
        tools
        and len(tools) > 0
        and model
        not in [
            "o1-mini",
            "o1-preview",
            "deepseek-chat",
            "deepseek-reasoner",
        ]
    ):
        function_specs = get_function_specs(tools, model)
        request_params["tools"] = function_specs
        if tool_choice:
            request_params["tool_choice"] = tool_choice
        
        if model in ["gpt-4o", "gpt-4o-mini"]:
            request_params["parallel_tool_calls"] = False

    # Some models do not allow temperature or response_format:
    if model.startswith("o") or model == "deepseek-reasoner":
        request_params.pop("temperature", None)
    if model in ["o1-mini", "o1-preview", "deepseek-chat", "deepseek-reasoner"]:
        request_params.pop("response_format", None)

    # Reasoning effort
    if model.startswith("o") and reasoning_effort is not None:
        request_params["reasoning_effort"] = reasoning_effort

    # Special case: model in ["gpt-4o", "gpt-4o-mini"] with `prediction`
    if model in ["gpt-4o", "gpt-4o-mini"] and prediction is not None:
        request_params["prediction"] = prediction
        request_params.pop("max_completion_tokens", None)
        request_params.pop("response_format", None)

    # Finally, set response_format if still relevant:
    if response_format:
        request_params["response_format"] = response_format
        # cannot have stop when using response_format
        request_params.pop("stop", None)

    return request_params


async def _process_openai_response(
    client,
    response,
    request_params,
    tools,
    tool_dict,
    response_format,
    model,
    is_async,
):
    """
    Extract content (including any tool calls) and usage info from OpenAI response.
    Handles chaining of tool calls.
    """
    if len(response.choices) == 0:
        raise Exception("No response from OpenAI.")
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")

    # If we have tools, handle dynamic chaining:
    tools_used = []
    if tools and len(tools) > 0:
        while True:
            message = response.choices[0].message
            if message.tool_calls:
                tool_call = message.tool_calls[0]
                func_name = tool_call.function.name
                try:
                    args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                try:
                    tool_to_call = tool_dict[func_name]
                except KeyError:
                    raise Exception(f"Tool `{func_name}` not found")

                # Execute tool depending on whether it is async
                try:
                    if inspect.iscoroutinefunction(tool_to_call):
                        result = await execute_tool_async(tool_to_call, args)
                    else:
                        result = execute_tool(tool_to_call, args)
                except Exception as e:
                    raise Exception(f"Error executing tool `{func_name}`: {e}")
                tools_used.append(func_name)

                # Append the tool calls as an assistant response
                request_params["messages"].append(
                    {
                        "role": "assistant",
                        "tool_calls": message.tool_calls,
                    }
                )

                # Append the tool message
                request_params["messages"].append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
                # Make next call
                if is_async:
                    response = await client.chat.completions.create(**request_params)
                else:
                    response = client.chat.completions.create(**request_params)
            else:
                content = message.content
                break
    else:
        # No tools provided
        if response_format and model not in ["o1-mini", "o1-preview"]:
            content = response.choices[0].message.parsed
        else:
            content = response.choices[0].message.content

    usage = response.usage
    return (
        content,
        tools_used,
        usage.prompt_tokens,
        usage.completion_tokens,
        usage.completion_tokens_details,
    )


def _process_openai_response_handler(
    client,
    response,
    request_params: Dict[str, Any],
    tools: List[Callable],
    tool_dict: Dict[str, Callable],
    response_format,
    model: str,
    is_async=False,
):
    """
    Processes OpenAI's response by determining whether to execute the response handling
    synchronously or asynchronously. This function acts as a wrapper around _process_openai_response,
    deciding the execution mode based on the is_async parameter.

    Parameters:
    - client: The client instance used for communication.
    - response: The response object from OpenAI.
    - request_params: A dictionary of request parameters that resulted in the response.
    - tools: A list of callable tools available for function calling.
    - tool_dict: A dictionary mapping tool names to their callable functions.
    - is_async: A boolean flag indicating whether to execute asynchronously.

    Returns:
    - The processed response content, input tokens, and output tokens from the response.
    """
    try:
        if is_async:
            return _process_openai_response(
                client=client,
                response=response,
                request_params=request_params,
                tools=tools,
                tool_dict=tool_dict,
                response_format=response_format,
                model=model,
                is_async=is_async,
            )  # Caller must await this
        else:
            return asyncio.run(
                _process_openai_response(
                    client=client,
                    response=response,
                    request_params=request_params,
                    tools=tools,
                    tool_dict=tool_dict,
                    response_format=response_format,
                    model=model,
                    is_async=is_async,
                )
            )

    except Exception as e:
        raise Exception("Error processing OpenAI response:", e)


def chat_openai(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: Union[OpenAIToolChoice, OpenAIForcedFunction] = None,
    base_url: str = "https://api.openai.com/v1/",
    api_key: str = os.environ.get("OPENAI_API_KEY", ""),
    prediction=None,
    reasoning_effort=None,
    store=True,
    metadata=None,
    timeout=100,
):
    """Synchronous OpenAI chat."""
    from openai import OpenAI

    t = time.time()
    client_openai = OpenAI(base_url=base_url, api_key=api_key)
    request_params = _build_openai_params(
        messages=messages,
        model=model,
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

    # Construct a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

    # If response_format is set, we do parse
    if request_params.get("response_format"):
        response = client_openai.beta.chat.completions.parse(**request_params)
    else:
        response = client_openai.chat.completions.create(**request_params)

    content, tools_used, prompt_tokens, output_tokens, completion_token_details = (
        _process_openai_response_handler(
            client=client_openai,
            response=response,
            request_params=request_params,
            tools=tools,
            tool_dict=tool_dict,
            response_format=response_format,
            model=model,
            is_async=False,
        )
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_tokens_details=completion_token_details,
        tools_used=tools_used,
    )


async def chat_openai_async(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    max_completion_tokens: int = 16384,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools: List[Callable] = None,
    tool_choice: Union[OpenAIToolChoice, OpenAIForcedFunction] = None,
    store=True,
    metadata=None,
    timeout=100,
    base_url: str = "https://api.openai.com/v1/",
    api_key: str = os.environ.get("OPENAI_API_KEY", ""),
    prediction: Dict[str, str] = None,
    reasoning_effort=None,
):
    """Asynchronous OpenAI chat."""
    from openai import AsyncOpenAI

    t = time.time()
    client_openai = AsyncOpenAI(base_url=base_url, api_key=api_key)
    request_params = _build_openai_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        seed=seed,
        tools=tools,
        tool_choice=tool_choice,
        prediction=prediction,
        reasoning_effort=reasoning_effort,
        store=store,
        metadata=metadata,
        timeout=timeout,
    )

    # Build a tool dict if needed
    tool_dict = {}
    if tools and len(tools) > 0 and "tools" in request_params:
        tool_dict = {tool.__name__: tool for tool in tools}

    # If response_format is set, we do parse
    if request_params.get("response_format"):
        response = await client_openai.beta.chat.completions.parse(**request_params)
    else:
        response = await client_openai.chat.completions.create(**request_params)

    content, tools_used, prompt_tokens, output_tokens, completion_token_details = (
        await _process_openai_response_handler(
            client=client_openai,
            response=response,
            request_params=request_params,
            tools=tools,
            tool_dict=tool_dict,
            response_format=response_format,
            model=model,
            is_async=True,
        )
    )

    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=prompt_tokens,
        output_tokens=output_tokens,
        output_tokens_details=completion_token_details,
        tools_used=tools_used,
    )


#
# --------------------------------------------------------------------------------
# 3) TOGETHER
# --------------------------------------------------------------------------------
#


def _build_together_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    seed: int = 0,
):
    return {
        "messages": messages,
        "model": model,
        "max_tokens": max_completion_tokens,
        "temperature": temperature,
        "stop": stop,
        "seed": seed,
    }


def _process_together_response(response):
    if response.choices[0].finish_reason == "length":
        raise Exception("Max tokens reached")
    if len(response.choices) == 0:
        raise Exception("Max tokens reached")
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def chat_together(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools=None,
    tool_choice=None,
):
    """Synchronous Together chat."""
    from together import Together

    t = time.time()
    client_together = Together()
    params = _build_together_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
    )
    response = client_together.chat.completions.create(**params)

    content, input_toks, output_toks = _process_together_response(response)
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
    )


async def chat_together_async(
    messages: List[Dict[str, str]],
    model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    max_completion_tokens: int = 4096,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools=None,
    tool_choice=None,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
):
    """Asynchronous Together chat."""
    from together import AsyncTogether

    t = time.time()
    client_together = AsyncTogether(timeout=timeout)
    params = _build_together_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        seed=seed,
    )
    response = await client_together.chat.completions.create(**params)

    content, input_toks, output_toks = _process_together_response(response)
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
    )


#
# --------------------------------------------------------------------------------
# 4) GEMINI
# --------------------------------------------------------------------------------
#


def _build_gemini_params(
    messages: List[Dict[str, str]],
    model: str,
    max_completion_tokens: int,
    temperature: float,
    stop: List[str],
    response_format=None,
    seed: int = 0,
    store=True,
    metadata=None,
):
    """Construct parameters for Gemini's generate_content call."""
    if messages[0]["role"] == "system":
        system_msg = messages[0]["content"]
        messages = messages[1:]
    else:
        system_msg = None

    # Combine all user/assistant messages into one string
    message = "\n".join([m["content"] for m in messages])
    config = {
        "temperature": temperature,
        "system_instruction": system_msg,
        "max_output_tokens": max_completion_tokens,
        "stop_sequences": stop,
    }

    if response_format:
        # If we want a JSON / Pydantic format
        # "response_schema" is only recognized if the google.genai library supports it
        config["response_mime_type"] = "application/json"
        config["response_schema"] = response_format

    return message, config


def _process_gemini_response(response, response_format=None):
    """Extract the response content & usage from Gemini result, optionally parse JSON."""
    content = response.text
    if response_format:
        # Attempt to parse with pydantic model
        content = response_format.model_validate_json(content)
    usage_meta = response.usage_metadata
    return content, usage_meta.prompt_token_count, usage_meta.candidates_token_count


def chat_gemini(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools=None,
    tool_choice=None,
    store=True,
    metadata=None,
):
    """Synchronous Gemini chat."""
    from google import genai
    from google.genai import types

    t = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    message, generation_cfg = _build_gemini_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        seed=seed,
        store=store,
        metadata=metadata,
    )

    try:
        response = client.models.generate_content(
            model=model,
            contents=message,
            config=types.GenerateContentConfig(**generation_cfg),
        )
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    content, input_toks, output_toks = _process_gemini_response(
        response, response_format
    )
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
    )


async def chat_gemini_async(
    messages: List[Dict[str, str]],
    model: str = "gemini-2.0-flash",
    max_completion_tokens: int = 8192,
    temperature: float = 0.0,
    stop: List[str] = [],
    response_format=None,
    seed: int = 0,
    tools=None,
    tool_choice=None,
    store=True,
    metadata=None,
    timeout=100,
    prediction=None,
    reasoning_effort=None,
):
    """Asynchronous Gemini chat."""
    from google import genai
    from google.genai import types

    t = time.time()
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", ""))
    message, generation_cfg = _build_gemini_params(
        messages=messages,
        model=model,
        max_completion_tokens=max_completion_tokens,
        temperature=temperature,
        stop=stop,
        response_format=response_format,
        seed=seed,
        store=store,
        metadata=metadata,
    )

    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=message,
            config=types.GenerateContentConfig(**generation_cfg),
        )
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

    content, input_toks, output_toks = _process_gemini_response(
        response, response_format
    )
    return LLMResponse(
        model=model,
        content=content,
        time=round(time.time() - t, 3),
        input_tokens=input_toks,
        output_tokens=output_toks,
    )
