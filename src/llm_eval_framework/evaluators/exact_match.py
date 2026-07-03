from typing import Any, Dict
from llm_eval_framework.evaluators.base import BaseEvaluator, EvaluationResult


class ExactMatchEvaluator(BaseEvaluator):
    def __init__(self, case_sensitive: bool = False) -> None:
        self.case_sensitive = case_sensitive

    def evaluate(self, *, expected: str, actual: str, metadata: Dict[str, Any]) -> EvaluationResult:
        expected_value = expected if self.case_sensitive else expected.lower()
        actual_value = actual if self.case_sensitive else actual.lower()
        passed = expected_value.strip() == actual_value.strip()
        return EvaluationResult(
            name="exact_match",
            passed=passed,
            score=1.0 if passed else 0.0,
            details={"expected": expected, "actual": actual},
        )
