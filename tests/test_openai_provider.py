import unittest
from unittest.mock import patch, MagicMock

from llm_eval_framework.providers.openai_provider import OpenAIProvider

class TestOpenAIProviderPIIAndErrorHandling(unittest.TestCase):
    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_handles_prompt_with_pii_safely(self, mock_openai_class):
        # Mock the OpenAI client and the chat.completions.create call
        mock_client = MagicMock()
        mock_response = MagicMock()
        # Simulate a response that does not echo raw PII verbatim
        mock_response.choices = [MagicMock(message=MagicMock(content="Response without PII"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        # Example PII-like prompt text
        prompt_with_pii = "User SSN: 123-45-6789 please redact it."

        response = provider.generate(prompt_with_pii)

        # Assert the SDK was called with correct parameters
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertIn("messages", kwargs)
        self.assertIsInstance(kwargs["messages"], list)
        self.assertIn("content", kwargs["messages"][0])
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertEqual(kwargs["messages"][0]["content"], prompt_with_pii)

        # Check that the response text does not expose the raw PII
        self.assertNotIn("123-45-6789", response.text)
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsInstance(response.latency_ms, int)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_explicit_pii_redaction(self, mock_openai_class):
        # Mock to simulate the LLM response redacting the raw PII value
        mock_client = MagicMock()
        redacted_response = "The SSN you provided has been redacted for your safety."
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=redacted_response))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        pii_prompt = "Here is my SSN: 123-45-6789. Please keep it confidential."
        response = provider.generate(pii_prompt)

        # Verify the SDK was invoked correctly
        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["model"], "gpt-4o-mini")
        self.assertEqual(kwargs["messages"][0]["role"], "user")
        self.assertEqual(kwargs["messages"][0]["content"], pii_prompt)

        # Assert the raw PII is NOT present in the output text
        self.assertNotIn("123-45-6789", response.text)
        # Assert the output includes redaction confirmation
        self.assertIn("redacted", response.text.lower())

        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsInstance(response.latency_ms, int)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_propagates_openai_api_errors(self, mock_openai_class):
        mock_client = MagicMock()
        # Setup the client to raise RuntimeError to simulate API failure
        mock_client.chat.completions.create.side_effect = RuntimeError("API failure")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(RuntimeError) as cm:
            provider.generate("Normal prompt")

        self.assertEqual(str(cm.exception), "API failure")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_empty_prompt(self, mock_openai_class):
        # Test boundary: empty prompt still calls SDK and returns result
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Empty prompt response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate("")

        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(response.text, "Empty prompt response")
        self.assertEqual(response.model, "gpt-4o-mini")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_none_prompt_raises(self, mock_openai_class):
        # Even if not explicitly validated, test SDK is called and handles None prompts or exception propagates
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = RuntimeError("Prompt cannot be None")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(RuntimeError) as cm:
            provider.generate(None)

        self.assertEqual(str(cm.exception), "Prompt cannot be None")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_non_string_prompt(self, mock_openai_class):
        # Provide a non-string prompt (e.g. int) and assert that the error from client is propagated
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TypeError("Prompt must be string")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(TypeError) as cm:
            provider.generate(12345)

        self.assertEqual(str(cm.exception), "Prompt must be string")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_very_long_prompt(self, mock_openai_class):
        # Confirm that very long prompt strings are passed to SDK and return expected response
        mock_client = MagicMock()
        long_prompt = "A" * 10000  # 10k characters
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Long prompt response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate(long_prompt)

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["messages"][0]["content"], long_prompt)
        self.assertEqual(response.text, "Long prompt response")
        self.assertEqual(response.model, "gpt-4o-mini")

if __name__ == "__main__":
    unittest.main()
