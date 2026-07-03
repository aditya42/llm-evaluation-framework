from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class EvaluationResult:
    name: str
    passed: bool
    score: float
    details: Dict[str, Any]


class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, *, expected: str, actual: str, metadata: Dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError
