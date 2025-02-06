import json
import unittest
import pytest
from ..defog_utils.utils_llm import (
    LLMResponse,
    chat_anthropic,
    chat_gemini,
    chat_openai,
    chat_together,
    chat_anthropic_async,
    chat_openai_async,
    chat_together_async,
)

messages_no_sys = [
    {"role": "user", "content": "Return a greeting in not more than 2 words\n"}
]
messages_sql = [
    {
        "role": "system",
        "content": "Your task is to generate SQL given a natural language question and schema of the user's database. Do not use aliases. Return only the SQL without ```.",
    },
    {
        "role": "user",
        "content": f"""Question: What is the total number of orders?
Schema:
```sql
CREATE TABLE orders (
    order_id int,
    customer_id int,
    employee_id int,
    order_date date
);
```
""",
    },
]

messages_json = [
    {
        "role": "system",
        "content": 'Your task is to generate SQL given a natural language question and schema of the user\'s database. Return your answer only as a JSON object with the reasoning in the \'reasoning\' field and SQL in the \'sql\' field, without ```. For example, {"sql": "...", "reasoning": "..."}',
    },
    {
        "role": "user",
        "content": f"""Question: What is the total number of orders?
Schema:
```sql
CREATE TABLE orders (
    order_id int,
    customer_id int,
    employee_id int,
    order_date date
);
```
""",
    },
]

acceptable_sql = [
    "select count(*) from orders",
    "select count(order_id) from orders",
    "select count(*) as total_orders from orders",
    "select count(order_id) as total_orders from orders",
]


class TestChatClients(unittest.TestCase):

    def check_sql(self, sql: str):
        self.assertIn(sql.strip(";\n").lower(), acceptable_sql)

    def test_chat_anthropic_no_sys(self):
        response = chat_anthropic(
            messages_no_sys,
            model="claude-3-haiku-20240307",
            max_completion_tokens=10,
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)
        self.assertEqual(response.input_tokens, 18)
        self.assertLessEqual(response.output_tokens, 10)

    @pytest.mark.asyncio
    async def test_chat_anthropic_no_sys_async(self):
        response = await chat_anthropic_async(
            "claude-3-haiku-20240307",
            messages_no_sys,
            max_completion_tokens=10,
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)
        self.assertEqual(response.input_tokens, 18)
        self.assertLessEqual(response.output_tokens, 10)

    def test_chat_gemini_no_sys(self):
        response = chat_gemini(
            messages_no_sys,
            model="gemini-1.5-flash",
            max_completion_tokens=10,
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)

    def test_chat_openai_no_sys(self):
        response = chat_openai(
            messages_no_sys,
            model="gpt-4o-mini",
            max_completion_tokens=10,
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)

    @pytest.mark.asyncio
    async def test_chat_openai_no_sys_async(self):
        response = await chat_openai_async(
            "gpt-4o-mini", messages_no_sys, max_completion_tokens=10, seed=0
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)
        self.assertEqual(response.input_tokens, 18)
        self.assertLessEqual(response.output_tokens, 10)

    def test_chat_together_no_sys(self):
        response = chat_together(
            messages_no_sys,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            max_completion_tokens=10,
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)
        self.assertEqual(response.input_tokens, 45)  # hidden sys prompt added I think
        self.assertLessEqual(response.output_tokens, 10)

    @pytest.mark.asyncio
    async def test_chat_together_no_sys_async(self):
        response = await chat_together_async(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages_no_sys,
            max_completion_tokens=10,
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIsInstance(response.content, str)
        self.assertEqual(response.input_tokens, 46)  # hidden sys prompt added I think
        self.assertLessEqual(response.output_tokens, 10)

    def test_chat_anthropic_sql(self):
        response = chat_anthropic(
            messages_sql,
            model="claude-3-haiku-20240307",
            max_completion_tokens=100,
            stop=[";"],
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertEqual(response.input_tokens, 90)  # 90 input tokens
        self.assertTrue(response.output_tokens < 15)  # output tokens should be < 15

    @pytest.mark.asyncio
    async def test_chat_anthropic_sql_async(self):
        response = await chat_anthropic_async(
            "claude-3-haiku-20240307",
            messages_sql,
            max_completion_tokens=100,
            stop=[";"],
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertEqual(response.input_tokens, 90)  # 90 input tokens
        self.assertTrue(response.output_tokens < 15)  # output tokens should be < 15

    def test_chat_openai_sql(self):
        response = chat_openai(messages_sql, model="gpt-4o-mini", stop=[";"], seed=0)
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertEqual(response.input_tokens, 83)
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    @pytest.mark.asyncio
    async def test_chat_openai_sql_async(self):
        response = await chat_openai_async(
            "gpt-4o-mini", messages_sql, stop=[";"], seed=0
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertEqual(response.input_tokens, 83)
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    def test_chat_together_sql(self):
        response = chat_together(
            messages_sql,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            stop=[";"],
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertEqual(response.input_tokens, 107)
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    @pytest.mark.asyncio
    async def test_chat_together_sql_async(self):
        response = await chat_together_async(
            "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            messages_sql,
            stop=[";"],
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertEqual(response.input_tokens, 108)
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    def test_chat_gemini_sql(self):
        response = chat_gemini(
            messages_sql, model="gemini-1.5-flash", stop=[";"], seed=0
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.check_sql(response.content)
        self.assertTrue(response.output_tokens < 10)