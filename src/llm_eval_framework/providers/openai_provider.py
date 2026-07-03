import os
import time
from openai import OpenAI
from llm_eval_framework.providers.base import BaseLLMProvider, LLMResponse


class OpenAIProvider(BaseLLMProvider):
    def __init__(self, model: str = "gpt-4o-mini") -> None:
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate(self, prompt: str) -> LLMResponse:
        start = time.perf_counter()
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, latency_ms=latency_ms, model=self.model)
