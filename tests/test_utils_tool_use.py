import unittest
import pytest
from ..defog_utils.utils_multi_llm import chat_async
from pydantic import BaseModel
import requests
import os

from bs4 import BeautifulSoup

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

DEFOG_API_KEY = os.environ.get("DEFOG_API_KEY")

if DEFOG_API_KEY is None:
    raise ValueError("DEFOG_API_KEY is not set, the search test cannot be run")

class TestToolUseFeatures(unittest.IsolatedAsyncioTestCase):
    @pytest.mark.asyncio
    async def test_tool_use_arithmetic(self):
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

        tools = [numsum, numprod]

        result = await chat_async(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "What is the product of 3 and 15, added to 5? Return only the final answer, nothing else.",},
            ],
            tools=tools,
        )
        self.assertEqual(result.content, '50')

    @pytest.mark.asyncio
    async def test_tool_use_search(self):
        class SearchInput(BaseModel):
            query: str = ""

        def search(input: SearchInput):
            """
            This function searches Google for the given query. It then visits the first result page, and returns the HTML content of the page.
            """
            r = requests.post(
                "https://api.defog.ai/unstructured_data/search",
                json={"api_key": DEFOG_API_KEY, "user_question": input.query},
            )
            first_result_link = r.json()["organic"][0]["link"]
            r = requests.get(first_result_link)
            return clean_html_text(r.text)

        tools = [search]

        result = await chat_async(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": "Who is the Prime Minister of Singapore right now (in 2025)? Recall that the current year is 2025. Return your answer as a single phrase.",},
            ],
            tools=tools,
        )
        self.assertIn('lawrence wong', result.content.lower())