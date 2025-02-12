import unittest
import pytest
from ..defog_utils.utils_multi_llm import chat_async, chat
from ..defog_utils.utils_llm import chat_anthropic, chat_openai
from pydantic import BaseModel
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
    query: str = ""


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
    This function return the sum of two numbers
    """
    return input.a + input.b


def numprod(input: Numbers):
    """
    This function return the product of two numbers
    """
    return input.a * input.b


# ==================================================================================================
# Tests
# ==================================================================================================
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
