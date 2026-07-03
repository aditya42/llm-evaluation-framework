from typing import Any, Dict
from llm_eval_framework.evaluators.base import BaseEvaluator, EvaluationResult


class LatencyCheckEvaluator(BaseEvaluator):
    def __init__(self, max_latency_ms: int = 3000) -> None:
        self.max_latency_ms = max_latency_ms

    def evaluate(self, *, expected: str, actual: str, metadata: Dict[str, Any]) -> EvaluationResult:
        latency_ms = int(metadata.get("latency_ms", 0))
        passed = latency_ms <= self.max_latency_ms
        score = 1.0 if passed else max(0.0, self.max_latency_ms / latency_ms)
        return EvaluationResult(
            name="latency_check",
            passed=passed,
            score=score,
            details={"latency_ms": latency_ms, "max_latency_ms": self.max_latency_ms},
        )
