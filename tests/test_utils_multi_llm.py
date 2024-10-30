import unittest
from ..defog_utils.utils_multi_llm import map_model_to_chat_fn, chat
from ..defog_utils.utils_llm import LLMResponse, chat_anthropic, chat_gemini, chat_openai, chat_together

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

acceptable_sql = [
    "select count(*) from orders",
    "select count(order_id) from orders",
    "select count(*) as total_orders from orders",
    "select count(order_id) as total_orders from orders",
]

class TestChatClients(unittest.TestCase):
    def check_sql(self, sql: str):
        self.assertIn(sql.strip(";\n").lower(), acceptable_sql)

    def test_map_model_to_chat_fn(self):
        self.assertEqual(map_model_to_chat_fn("claude-3-5-sonnet-20241022"), chat_anthropic)
        self.assertEqual(map_model_to_chat_fn("gemini-1.5-pro"), chat_gemini)
        self.assertEqual(map_model_to_chat_fn("gpt-4o"), chat_openai)
        self.assertEqual(map_model_to_chat_fn("mistralai/Mistral-7B-Instruct-v0.3"), chat_together)
        self.assertEqual(map_model_to_chat_fn("meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"), chat_together)
        self.assertEqual(map_model_to_chat_fn("Qwen/Qwen2.5-72B-Instruct-Turbo"), chat_together)
        with self.assertRaises(ValueError):
            map_model_to_chat_fn("unknown-model")

    def test_simple_chat(self):
        models = ["claude-3-haiku-20240307", "gemini-1.5-flash-002", "gpt-4o-mini", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"]
        messages = [{"role": "user", "content": "Return a greeting in not more than 2 words\n"}]
        responses = chat(models, messages, max_tokens=20, temperature=0.0, stop=[";"], json_mode=False, seed=0)
        self.assertIsInstance(responses, dict)
        for model in models:
            self.assertIn(model, responses)
            response = responses[model]
            print(model, response)
            self.assertIsInstance(response, LLMResponse)
            self.assertIsInstance(response.content, str)
            self.assertIsInstance(response.time, float)
            self.assertLess(response.input_tokens, 50) # higher as default system prompt is added in together's API when none provided
            self.assertLess(response.output_tokens, 20)

    def test_sql_chat(self):
        models = ["claude-3-haiku-20240307", "gemini-1.5-flash-002", "gpt-4o-mini", "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"]
        responses = chat(models, messages_sql, max_tokens=20, temperature=0.0, stop=[";"], json_mode=False, seed=0)
        self.assertIsInstance(responses, dict)
        for model in models:
            self.assertIn(model, responses)
            response = responses[model]
            print(model, response)
            self.assertIsInstance(response, LLMResponse)
            self.assertIsInstance(response.content, str)
            self.assertIsInstance(response.time, float)
            self.assertLess(response.input_tokens, 110)
            self.assertLess(response.output_tokens, 20)

if __name__ == "__main__":
    unittest.main()