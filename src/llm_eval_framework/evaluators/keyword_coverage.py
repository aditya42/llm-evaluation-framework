from typing import Any, Dict, List
from llm_eval_framework.evaluators.base import BaseEvaluator, EvaluationResult


class KeywordCoverageEvaluator(BaseEvaluator):
    def __init__(self, min_score: float = 0.7, case_sensitive: bool = False) -> None:
        self.min_score = min_score
        self.case_sensitive = case_sensitive

    def evaluate(self, *, expected: str, actual: str, metadata: Dict[str, Any]) -> EvaluationResult:
        keywords: List[str] = metadata.get("required_keywords", [])
        if not keywords:
            return EvaluationResult("keyword_coverage", True, 1.0, {"reason": "No keywords required"})

        actual_text = actual if self.case_sensitive else actual.lower()
        normalized_keywords = keywords if self.case_sensitive else [kw.lower() for kw in keywords]
        matched = [kw for kw in normalized_keywords if kw in actual_text]
        score = len(matched) / len(normalized_keywords)

        return EvaluationResult(
            name="keyword_coverage",
            passed=score >= self.min_score,
            score=score,
            details={
                "required_keywords": keywords,
                "matched_keywords": matched,
                "min_score": self.min_score,
            },
        )
