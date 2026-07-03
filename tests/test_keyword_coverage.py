from llm_eval_framework.evaluators.keyword_coverage import KeywordCoverageEvaluator


def test_keyword_coverage_passes_when_required_keywords_are_present():
    evaluator = KeywordCoverageEvaluator(min_score=0.5)
    result = evaluator.evaluate(
        expected="",
        actual="Regression testing validates existing functionality after changes.",
        metadata={"required_keywords": ["regression", "existing functionality"]},
    )
    assert result.passed is True
    assert result.score == 1.0


def test_keyword_coverage_fails_when_score_is_below_threshold():
    evaluator = KeywordCoverageEvaluator(min_score=0.8)
    result = evaluator.evaluate(
        expected="",
        actual="Regression testing validates old features.",
        metadata={"required_keywords": ["regression", "existing functionality", "after changes"]},
    )
    assert result.passed is False
