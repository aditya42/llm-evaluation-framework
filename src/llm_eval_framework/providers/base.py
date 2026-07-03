from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class LLMResponse:
    text: str
    latency_ms: int
    model: str


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> LLMResponse:
        raise NotImplementedError
