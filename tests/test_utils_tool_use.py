import unittest
import pytest
from ..defog_utils.utils_multi_llm import chat_async, chat
from ..defog_utils.utils_llm import chat_anthropic, chat_openai
from ..defog_utils.utils_function_calling import get_function_specs
from pydantic import BaseModel, Field
import httpx
import os

from bs4 import BeautifulSoup

DEFOG_API_KEY = os.environ.get("DEFOG_API_KEY")

if DEFOG_API_KEY is None:
    raise ValueError("DEFOG_API_KEY is not set, the search test cannot be run")

# ==================================================================================================
# Functions for function calling
# ==================================================================================================


def clean_html_text(html_text):
    """
    Remove HTML tags from the given HTML text and return plain text.

    Args:
        html_text (str): A string containing HTML content.

    Returns:
        str: A string with the HTML tags removed.
    """
    # Parse the HTML content
    soup = BeautifulSoup(html_text, "html.parser")

    # Extract text from the parsed HTML
    # The separator parameter defines what string to insert between text blocks.
    # strip=True removes leading and trailing whitespace from each piece of text.
    cleaned_text = soup.get_text(separator=" ", strip=True)

    return cleaned_text


class SearchInput(BaseModel):
    query: str = Field(default="", description="The query to search for")


async def search(input: SearchInput):
    """
    This function searches Google for the given query. It then visits the first result page, and returns the HTML content of the page.
    """
    async with httpx.AsyncClient() as client:
        r = await client.post(
            "https://api.defog.ai/unstructured_data/search",
            json={"api_key": DEFOG_API_KEY, "user_question": input.query},
        )
        first_result_link = r.json()["organic"][0]["link"]
        r = await client.get(first_result_link)
    return clean_html_text(r.text)


class Numbers(BaseModel):
    a: int = 0
    b: int = 0


def numsum(input: Numbers):
    """
    This function returns the sum of two numbers
    """
    return input.a + input.b


def numprod(input: Numbers):
    """
    This function returns the product of two numbers
    """
    return input.a * input.b


# ==================================================================================================
# Tests
# ==================================================================================================
class TestGetFunctionSpecs(unittest.TestCase):
    def setUp(self):
        self.openai_model = "gpt-4o"
        self.anthropic_model = "claude-3-haiku-20240307"
        self.tools = [search, numsum, numprod]
        self.maxDiff = None
        self.openai_specs = [
            {
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "This function searches Google for the given query. It then visits the first result page, and returns the HTML content of the page.",
                    "parameters": {
                        "properties": {
                            "query": {"default": "", "description": "The query to search for", "title": "Query", "type": "string"}
                        },
                        "title": "SearchInput",
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "numsum",
                    "description": "This function returns the sum of two numbers",
                    "parameters": {
                        "properties": {
                            "a": {"default": 0, "title": "A", "type": "integer"},
                            "b": {"default": 0, "title": "B", "type": "integer"},
                        },
                        "title": "Numbers",
                        "type": "object",
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "numprod",
                    "description": "This function returns the product of two numbers",
                    "parameters": {
                        "properties": {
                            "a": {"default": 0, "title": "A", "type": "integer"},
                            "b": {"default": 0, "title": "B", "type": "integer"},
                        },
                        "title": "Numbers",
                        "type": "object",
                    },
                },
            },
        ]
        self.anthropic_specs = [
            {
                "name": "search",
                "description": "This function searches Google for the given query. It then visits the first result page, and returns the HTML content of the page.",
                "input_schema": {
                    "properties": {
                        "query": {"default": "", "description": "The query to search for", "title": "Query", "type": "string"}
                    },
                    "title": "SearchInput",
                    "type": "object",
                },
            },
            {
                "name": "numsum",
                "description": "This function returns the sum of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"default": 0, "title": "A", "type": "integer"},
                        "b": {"default": 0, "title": "B", "type": "integer"},
                    },
                    "title": "Numbers",
                    "type": "object",
                },
            },
            {
                "name": "numprod",
                "description": "This function returns the product of two numbers",
                "input_schema": {
                    "properties": {
                        "a": {"default": 0, "title": "A", "type": "integer"},
                        "b": {"default": 0, "title": "B", "type": "integer"},
                    },
                    "title": "Numbers",
                    "type": "object",
                },
            },
        ]

    def test_get_function_specs(self):
        openai_specs = get_function_specs(self.tools, self.openai_model)
        anthropic_specs = get_function_specs(self.tools, self.anthropic_model)

        self.assertEqual(openai_specs, self.openai_specs)
        self.assertEqual(anthropic_specs, self.anthropic_specs)


class TestToolUseFeatures(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tools = [search, numsum, numprod]
        self.search_qn = "Who is the Prime Minister of Singapore right now (in 2025)? Recall that the current year is 2025. Return your answer as a single phrase."
        self.search_answer = "lawrence wong"

        self.arithmetic_qn = "What is the product of 31283 and 2323, added to 5? Return only the final answer, nothing else."
        self.arithmetic_answer = "72670414"

    @pytest.mark.asyncio
    async def test_tool_use_arithmetic_async_openai(self):
        tools = self.tools

        result = await chat_async(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=tools,
        )
        self.assertEqual(result.content, self.arithmetic_answer)
        self.assertSetEqual(set(result.tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    async def test_tool_use_search_async_openai(self):
        tools = self.tools
        result = await chat_async(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.search_qn,
                },
            ],
            tools=tools,
            max_retries=1,
        )
        self.assertIn(self.search_answer, result.content.lower())
        self.assertSetEqual(set(result.tools_used), {"search"})

    @pytest.mark.asyncio
    async def test_tool_use_arithmetic_async_anthropic(self):
        tools = self.tools

        result = await chat_async(
            model="claude-3-haiku-20240307",
            messages=[
                {
                    "role": "user",
                    "content": self.arithmetic_qn,
                },
            ],
            tools=tools,
        )
        self.assertEqual(result.content, self.arithmetic_answer)
        self.assertSetEqual(set(result.tools_used), {"numsum", "numprod"})

    @pytest.mark.asyncio
    async def test_tool_use_search_async_anthropic(self):
        tools = self.tools
        result = await chat_async(
            model="claude-3-haiku-20240307",
            messages=[
                {
                    "role": "user",
                    "content": self.search_qn,
                },
            ],
            tools=tools,
            max_retries=1,
        )
        self.assertIn(self.search_answer, result.content.lower())
        self.assertSetEqual(set(result.tools_used), {"search"})

    def test_async_tool_in_sync_function_openai(self):
        tools = self.tools
        result_openai = chat_openai(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": self.search_qn,
                },
            ],
            tools=tools,
        )
        self.assertIn(self.search_answer, result_openai.content.lower())
        self.assertSetEqual(set(result_openai.tools_used), {"search"})

    def test_async_tool_in_sync_function_anthropic(self):
        tools = self.tools
        result_anthropic = chat_anthropic(
            model="claude-3-5-sonnet-20241022",
            messages=[
                {
                    "role": "user",
                    "content": self.search_qn,
                },
            ],
            tools=tools,
        )
        self.assertIn(self.search_answer, result_anthropic.content.lower())
        self.assertSetEqual(set(result_anthropic.tools_used), {"search"})
