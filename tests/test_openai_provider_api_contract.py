import pytest
from unittest.mock import patch, MagicMock

from llm_eval_framework.providers.openai_provider import OpenAIProvider


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_successful_call(mock_openai):
    # Setup the mock OpenAI client and its response
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    prompt = "Hello world"
    response = provider.generate(prompt)

    # Assert that the client's chat.completions.create was called with expected args
    mock_client.chat.completions.create.assert_called_once()
    call_args, call_kwargs = mock_client.chat.completions.create.call_args
    assert call_kwargs["model"] == provider.model
    messages = call_kwargs["messages"]
    assert isinstance(messages, list)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == prompt
    assert call_kwargs["temperature"] == 0

    # Assert the returned response fields
    assert hasattr(response, "text")
    assert response.text == "Test response"
    assert response.model == provider.model
    assert isinstance(response.latency_ms, int)


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_propagates_sdk_exceptions(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    # Simulate an API error raised from chat.completions.create
    mock_client.chat.completions.create.side_effect = RuntimeError("API failure")

    provider = OpenAIProvider()
    prompt = "Some prompt"

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate(prompt)
    assert "API failure" in str(exc_info.value)


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_with_empty_string_prompt(mock_openai):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Empty prompt response"
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    prompt = ""
    response = provider.generate(prompt)

    # Confirm the call used the empty prompt string
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["content"] == prompt
    assert response.text == "Empty prompt response"


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_with_none_prompt_raises_type_error(mock_openai):
    # Since the provider passes prompt to OpenAI SDK, passing None should raise a TypeError inside the SDK call
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = TypeError("Prompt must be a string")

    provider = OpenAIProvider()
    prompt = None

    with pytest.raises(TypeError) as exc_info:
        provider.generate(prompt)
    assert "Prompt must be a string" in str(exc_info.value)


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_with_very_large_prompt_passes(mock_openai):
    # Test handling of a very large prompt string, assuming provider accepts and passes it
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Large prompt response"
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    # Create a large prompt string (e.g., 100_000 characters)
    prompt = "a" * 100_000
    response = provider.generate(prompt)

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["content"] == prompt
    assert response.text == "Large prompt response"


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_with_non_string_prompt_raises_type_error(mock_openai):
    # Passing a non-string prompt that is not None (e.g., int) should raise on the SDK call
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = TypeError("Prompt must be a string")

    provider = OpenAIProvider()
    prompt = 12345

    with pytest.raises(TypeError) as exc_info:
        provider.generate(prompt)
    assert "Prompt must be a string" in str(exc_info.value)


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_with_pii_containing_prompt(mock_openai):
    # Simulate a prompt containing PII, e.g., an email and phone number
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Response to PII prompt"
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider()
    prompt = "User email: user@example.com, phone: +1-202-555-0191"
    response = provider.generate(prompt)

    # Assert the client call uses the prompt as is
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["content"] == prompt

    # Assert response matches mocked
    assert response.text == "Response to PII prompt"
    assert response.model == provider.model
    assert isinstance(response.latency_ms, int)


@patch("llm_eval_framework.providers.openai_provider.OpenAI")
def test_generate_error_recovery_propagation(mock_openai):
    # Simulate an unexpected error during the API call to ensure proper error propagation
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.side_effect = RuntimeError("Simulated API error for recovery test")

    provider = OpenAIProvider()
    prompt = "Test prompt for error recovery"

    with pytest.raises(RuntimeError) as exc_info:
        provider.generate(prompt)

    assert "Simulated API error for recovery test" in str(exc_info.value)
