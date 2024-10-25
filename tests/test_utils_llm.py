import json
import unittest
from ..defog_utils.utils_llm import (
    LLMResponse,
    chat_anthropic,
    chat_openai,
    chat_together,
)

messages = [
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
    "SELECT COUNT(*) FROM orders",
    "SELECT COUNT(order_id) FROM orders",
]

acceptable_sql_from_json = set(
    [
        "SELECT COUNT(order_id) as total_orders FROM orders;",
        "SELECT COUNT(*) AS total_orders FROM orders;",
        "SELECT COUNT(order_id) FROM orders;",
        "SELECT COUNT(*) FROM orders;",
    ]
)


class TestChatClients(unittest.TestCase):
    def test_chat_anthropic(self):
        response = chat_anthropic(
            messages,
            model="claude-3-haiku-20240307",
            max_tokens=100,
            stop=[";"],
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIn(response.content, acceptable_sql)
        self.assertEqual(response.input_tokens, 90)  # 90 input tokens
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    def test_chat_openai(self):
        response = chat_openai(messages, model="gpt-4o-mini", stop=[";"], seed=0)
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIn(response.content, acceptable_sql)
        self.assertEqual(response.input_tokens, 83)
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    def test_chat_together(self):
        response = chat_together(
            messages,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            stop=[";"],
            seed=0,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        self.assertIn(response.content, acceptable_sql)
        self.assertEqual(response.input_tokens, 108)
        self.assertTrue(response.output_tokens < 10)  # output tokens should be < 10

    def test_chat_json_anthropic(self):
        response = chat_anthropic(
            messages_json,
            model="claude-3-haiku-20240307",
            max_tokens=100,
            seed=0,
            json_mode=True,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        resp_dict = json.loads(response.content)
        self.assertIn(resp_dict["sql"], acceptable_sql_from_json)
        self.assertIsInstance(resp_dict["reasoning"], str)
        self.assertIsInstance(response.input_tokens, int)
        self.assertIsInstance(response.output_tokens, int)

    def test_chat_json_openai(self):
        response = chat_openai(
            messages_json, model="gpt-4o-mini", seed=0, json_mode=True
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        resp_dict = json.loads(response.content)
        self.assertIn(resp_dict["sql"], acceptable_sql_from_json)
        self.assertIsInstance(resp_dict["reasoning"], str)
        self.assertIsInstance(response.input_tokens, int)
        self.assertIsInstance(response.output_tokens, int)

    def test_chat_json_together(self):
        response = chat_together(
            messages_json,
            model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
            seed=0,
            json_mode=True,
        )
        print(response)
        self.assertIsInstance(response, LLMResponse)
        raw_output = response.content
        resp_dict = json.loads(raw_output)
        self.assertIn(resp_dict["sql"], acceptable_sql_from_json)
        self.assertIsInstance(resp_dict["reasoning"], str)
        self.assertIsInstance(response.input_tokens, int)
        self.assertIsInstance(response.output_tokens, int)
