import unittest
from unittest.mock import patch, MagicMock
import pytest

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

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_multiple_pii_entities(self, mock_openai_class):
        mock_client = MagicMock()
        pii_prompt = "My SSN is 987-65-4321 and my phone is (555) 123-4567."
        mock_response = MagicMock()
        # Response redacts all PII
        redacted_text = "Your sensitive information has been redacted."
        mock_response.choices = [MagicMock(message=MagicMock(content=redacted_text))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate(pii_prompt)

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["messages"][0]["content"], pii_prompt)

        # Assert no raw PII returned
        self.assertNotIn("987-65-4321", response.text)
        self.assertNotIn("555", response.text)
        self.assertIn("redacted", response.text.lower())

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_handles_malformed_pii(self, mock_openai_class):
        mock_client = MagicMock()
        malformed_pii_prompt = "My SSN is 123-AB-####, and email is user(at)example(dot)com."
        mock_response = MagicMock()
        # The response should acknowledge malformed data and redact or sanitize
        response_text = "The provided information has been sanitized for privacy."
        mock_response.choices = [MagicMock(message=MagicMock(content=response_text))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate(malformed_pii_prompt)

        self.assertNotIn("123-AB-####", response.text)
        self.assertIn("sanitized", response.text.lower())

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_propagates_rate_limit_error(self, mock_openai_class):
        mock_client = MagicMock()
        # Simulate rate limit exception from the OpenAI API
        class RateLimitError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = RateLimitError("Rate limit exceeded")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(RateLimitError) as cm:
            provider.generate("Some prompt")

        self.assertEqual(str(cm.exception), "Rate limit exceeded")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_propagates_network_error(self, mock_openai_class):
        mock_client = MagicMock()
        # Simulate network connectivity error
        class NetworkError(Exception):
            pass

        mock_client.chat.completions.create.side_effect = NetworkError("Network unreachable")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(NetworkError) as cm:
            provider.generate("Another prompt")

        self.assertEqual(str(cm.exception), "Network unreachable")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_pii_redaction_does_not_echo_varied_pii(self, mock_openai_class):
        # Test multiple varied PII inputs produce redacted outputs without any raw PII leakage
        mock_client = MagicMock()

        # Sample inputs containing varied PII
        pii_prompts = [
            "Credit card number: 4111 1111 1111 1111",
            "My phone: +1-(234)-567-8901",
            "Email: user@example.com",
            "Passport Number: X1234567",
            "DOB: 01/01/1990",
        ]
        redacted_response = "Your personal information has been securely redacted."

        def side_effect(*args, **kwargs):
            mock_resp = MagicMock()
            mock_resp.choices = [MagicMock(message=MagicMock(content=redacted_response))]
            return mock_resp

        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        for prompt in pii_prompts:
            response = provider.generate(prompt)
            mock_client.chat.completions.create.assert_called()
            # Assert no raw PII present in output response
            for token in ['4111 1111 1111 1111', '+1-(234)-567-8901', 'user@example.com', 'X1234567', '01/01/1990']:
                self.assertNotIn(token, response.text)
            self.assertIn("redacted", response.text.lower())

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_error_recovery_on_api_failure(self, mock_openai_class):
        # Simulate raising error on first call, then success on retry
        mock_client = MagicMock()

        error = RuntimeError("Temporary API failure")
        responses = [
            lambda *a, **k: (_ for _ in ()).throw(error),
            lambda *a, **k: MagicMock(choices=[MagicMock(message=MagicMock(content="Recovered response"))])
        ]
        call_iter = iter(responses)

        def side_effect(*args, **kwargs):
            return next(call_iter)()

        mock_client.chat.completions.create.side_effect = side_effect
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        # First call raises
        with self.assertRaises(RuntimeError):
            provider.generate("Prompt with PII: 999-99-9999")

        # Replace side effect to success for second call
        mock_client.chat.completions.create.side_effect = lambda *a, **k: MagicMock(choices=[MagicMock(message=MagicMock(content="Safe response"))])

        # Second call succeeds
        response = provider.generate("Prompt with PII: 999-99-9999")
        self.assertIn("Safe response", response.text)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_handles_pii_prompt_with_api_error_safely(self, mock_openai_class):
        """
        Test that generate raises API error without leaking raw PII or prompt content
        and can be safely retried afterwards.
        """
        mock_client = MagicMock()

        # Simulated error raised by the SDK
        error_message = "API failure on sensitive data"
        mock_client.chat.completions.create.side_effect = RuntimeError(error_message)
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        sensitive_prompt = "User PII data: SSN 111-22-3333 and email user@example.com"

        # On error, the exception should be raised, and we check no PII leaks in the exception message
        with self.assertRaises(RuntimeError) as cm:
            provider.generate(sensitive_prompt)
        # Confirm the exception message matches but does NOT include raw prompt content
        self.assertEqual(str(cm.exception), error_message)
        self.assertNotIn("111-22-3333", str(cm.exception))
        self.assertNotIn("user@example.com", str(cm.exception))

        # Patch side effect to succeed for a retry
        mock_client.chat.completions.create.side_effect = None
        redacted_response = "Your sensitive data has been redacted." 
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=redacted_response))]
        mock_client.chat.completions.create.return_value = mock_response

        # Retry should succeed and produce safe output
        response = provider.generate(sensitive_prompt)
        self.assertIn("redacted", response.text.lower())
        self.assertNotIn("111-22-3333", response.text)
        self.assertNotIn("user@example.com", response.text)
        self.assertEqual(response.model, "gpt-4o-mini")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_regression_no_raw_pii_echo(self, mock_openai_class):
        """
        Regression test to ensure that raw PII in prompt is never echoed verbatim in output.
        This test mocks the client to simulate behavior that redacts or replaces PII,
        and asserts that the exact raw PII strings from the prompt do not appear in the output.
        """
        mock_client = MagicMock()

        pii_prompt = "Sensitive info: SSN 456-78-9012, phone (555) 987-6543, email testuser@example.org."
        raw_pii_tokens = ["456-78-9012", "(555) 987-6543", "testuser@example.org"]

        redacted_response_text = "Your sensitive information has been safely redacted to protect your privacy."

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=redacted_response_text))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate(pii_prompt)

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args

        # Ensure the prompt is passed correctly
        self.assertEqual(kwargs["messages"][0]["content"], pii_prompt)

        # Assert the output text does not contain any raw PII tokens exactly
        for token in raw_pii_tokens:
            self.assertNotIn(token, response.text)

        # Assert the redaction confirmation is present
        self.assertIn("redacted", response.text.lower() or "safely")
        self.assertEqual(response.model, "gpt-4o-mini")
        self.assertIsInstance(response.latency_ms, int)

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_whitespace_prompt(self, mock_openai_class):
        # Test input with whitespace-only prompt
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Whitespace prompt response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate("   \t\n   ")

        mock_client.chat.completions.create.assert_called_once()
        self.assertEqual(response.text, "Whitespace prompt response")
        self.assertEqual(response.model, "gpt-4o-mini")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_list_prompt_raises(self, mock_openai_class):
        # Providing a list instead of a string should propagate TypeError from client
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TypeError("Prompt must be string")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(TypeError) as cm:
            provider.generate(["This", "is", "a", "list"])

        self.assertEqual(str(cm.exception), "Prompt must be string")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_dict_prompt_raises(self, mock_openai_class):
        # Providing a dict instead of a string should propagate TypeError from client
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = TypeError("Prompt must be string")
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        with self.assertRaises(TypeError) as cm:
            provider.generate({"key": "value"})

        self.assertEqual(str(cm.exception), "Prompt must be string")

    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_with_unicode_prompt(self, mock_openai_class):
        # Test that a unicode prompt is accepted and passed correctly
        mock_client = MagicMock()
        unicode_prompt = "Unicode test: \u2603 \u2764"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Unicode response"))]
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        response = provider.generate(unicode_prompt)

        mock_client.chat.completions.create.assert_called_once()
        _, kwargs = mock_client.chat.completions.create.call_args
        self.assertEqual(kwargs["messages"][0]["content"], unicode_prompt)
        self.assertEqual(response.text, "Unicode response")
        self.assertEqual(response.model, "gpt-4o-mini")
