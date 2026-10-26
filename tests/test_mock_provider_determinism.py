import pytest
from llm_eval_framework.providers.mock_provider import MockProvider


def test_mockprovider_deterministic_outputs():
    provider = MockProvider()
    prompts = [
        "Explain CI/CD in detail.",
        "What is regression testing?",
        "Describe hallucination in LLMs.",
        "Give a generic response."
    ]

    # Store results from first run
    first_run_results = {}
    for prompt in prompts:
        response = provider.generate(prompt)
        first_run_results[prompt] = response

    # Call again and assert exact equality for each prompt response
    for prompt in prompts:
        response2 = provider.generate(prompt)
        response1 = first_run_results[prompt]
        # Check text and model fields are identical
        assert response2.text == response1.text
        assert response2.model == response1.model
        # latency_ms should be int and non-negative
        assert isinstance(response2.latency_ms, int) and response2.latency_ms >= 0


def test_mockprovider_response_schema_and_types():
    provider = MockProvider()
    prompt = "Test prompt schema validation"

    response = provider.generate(prompt)

    # Validate response object attributes
    assert hasattr(response, "text"), "Response should have 'text' attribute"
    assert hasattr(response, "latency_ms"), "Response should have 'latency_ms' attribute"
    assert hasattr(response, "model"), "Response should have 'model' attribute"

    # Validate types
    assert isinstance(response.text, str), "Response.text should be a string"
    assert isinstance(response.latency_ms, int), "Response.latency_ms should be an integer"
    assert response.latency_ms >= 0, "Response.latency_ms should be non-negative"
    assert isinstance(response.model, str), "Response.model should be a string"
    # The fixed model string for MockProvider is 'mock-provider'
    assert response.model == "mock-provider"


def test_mockprovider_repeated_calls_stability():
    provider = MockProvider()
    prompt = "Repeat stability check"

    outputs = [provider.generate(prompt).text for _ in range(5)]
    # All outputs must be identical
    assert all(output == outputs[0] for output in outputs)


def test_mockprovider_regression_response_schema_consistency():
    provider = MockProvider()
    prompts = ["Example prompt 1", "Example prompt 2", "Example prompt 3"]

    # Cache first responses for each prompt
    baseline_responses = {}
    for prompt in prompts:
        resp = provider.generate(prompt)
        baseline_responses[prompt] = resp
        # Check schema and types on baseline
        assert isinstance(resp.text, str), "Response.text must be string"
        assert isinstance(resp.model, str), "Response.model must be string"
        assert resp.model == "mock-provider", "Response.model must be 'mock-provider'"
        assert isinstance(resp.latency_ms, int), "Response.latency_ms must be int"
        assert resp.latency_ms >= 0, "Response.latency_ms must be non-negative"

    # Repeat calls multiple times to assert strict deterministic consistency
    for _ in range(10):
        for prompt in prompts:
            resp = provider.generate(prompt)
            baseline = baseline_responses[prompt]
            assert resp.text == baseline.text, f"Text mismatch for prompt '{prompt}'"
            assert resp.model == baseline.model, f"Model mismatch for prompt '{prompt}'"
            assert isinstance(resp.latency_ms, int) and resp.latency_ms >= 0, "Latency_ms must be non-negative int"
            # latency_ms can vary but should never be negative or non-int
