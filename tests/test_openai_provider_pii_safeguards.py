import unittest
from unittest.mock import patch, MagicMock

from llm_eval_framework.providers.openai_provider import OpenAIProvider


class TestOpenAIProviderPIISafeguards(unittest.TestCase):
    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_no_pii_success(self, mock_openai):
        # Setup a nominal successful response without PII
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="Hello, how can I assist you?"))]
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        prompt = "Greet me politely."
        response = provider.generate(prompt)

        self.assertIn("Hello", response.text)
        self.assertEqual(response.model, "gpt-4o-mini")
        # Latency should be int and positive
        self.assertIsInstance(response.latency_ms, int)
        self.assertGreaterEqual(response.latency_ms, 0)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_embedded_pii(self, mock_openai):
        # Simulate API response containing PII such as email and SSN
        pii_text = "Please contact john.doe@example.com or call 555-12-3456."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=pii_text))]
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = "Provide contact info with PII."

        response = provider.generate(prompt)

        # The provider returns the text as is (no automatic redaction in production)
        self.assertEqual(response.text, pii_text)
        self.assertIn("john.doe@example.com", response.text)
        self.assertIn("555-12-3456", response.text)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_empty_response(self, mock_openai):
        # Simulate empty string response including no PII
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = "Empty response test"

        response = provider.generate(prompt)

        self.assertEqual(response.text, "")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_explicit_pii_in_prompt(self, mock_openai):
        # Prompt contains explicit PII data
        response_text = "Received your SSN securely. Thank you."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=response_text))]
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = "My SSN is 123-45-6789. Please process securely."

        response = provider.generate(prompt)

        self.assertEqual(response.text, response_text)
        self.assertIn("SSN", response.text)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_malformed_prompt(self, mock_openai):
        # Simulate handling of a malformed prompt (e.g., non-UTF8 characters)
        response_text = "Error: Received malformed input."
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=response_text))]
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = "\udce2\udce2\udce2"  # Invalid Unicode surrogate

        response = provider.generate(prompt)

        self.assertEqual(response.text, response_text)
        self.assertIn("Error", response.text)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_api_raises_exception(self, mock_openai):
        # Simulate API client throwing an exception (e.g., due to bad input or network error)
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("API failure")
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = "Prompt that causes API failure"

        with self.assertRaises(RuntimeError) as context:
            provider.generate(prompt)

        self.assertIn("API failure", str(context.exception))

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_api_redacted_pii_response(self, mock_openai):
        # Simulate API response with PII redacted text
        redacted_text = "[REDACTED PERSONAL INFORMATION]"
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=redacted_text))]
        )
        mock_openai.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")
        prompt = "Provide sensitive info."

        response = provider.generate(prompt)

        self.assertEqual(response.text, redacted_text)
        self.assertTrue(response.text.startswith("[REDACTED"))


if __name__ == "__main__":
    unittest.main()
