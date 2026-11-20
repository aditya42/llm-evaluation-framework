import unittest
from unittest.mock import patch, MagicMock

from llm_eval_framework.providers.openai_provider import OpenAIProvider


class TestOpenAIProviderDeterministicGeneration(unittest.TestCase):
    @patch("llm_eval_framework.providers.openai_provider.OpenAI")
    def test_generate_deterministic_output(self, mock_openai_class):
        # Arrange
        fake_response_text = "This is a fixed deterministic response."
        prompt = "Deterministic test prompt."

        # Setup the mock OpenAI client instance and its chat.completions.create
        mock_client = MagicMock()

        # Create nested MagicMocks to simulate attribute access (response.choices[0].message.content)
        mock_message = MagicMock()
        mock_message.content = fake_response_text

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]

        mock_client.chat.completions.create.return_value = mock_response
        mock_openai_class.return_value = mock_client

        provider = OpenAIProvider(model="gpt-4o-mini")

        # Act
        response1 = provider.generate(prompt)
        response2 = provider.generate(prompt)

        # Assert
        self.assertEqual(response1.text, fake_response_text)
        self.assertEqual(response1.model, "gpt-4o-mini")
        self.assertEqual(response2.text, fake_response_text)
        self.assertEqual(response2.model, "gpt-4o-mini")

        # Assert that the underlying OpenAI client was called with correct parameters
        expected_call_args = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        mock_client.chat.completions.create.assert_called_with(**expected_call_args)

        # Assert responses are identical
        self.assertEqual(response1.text, response2.text)
        self.assertEqual(response1.model, response2.model)
