import argparse
from typing import Any, Dict, List

import pandas as pd

from llm_eval_framework.evaluators.keyword_coverage import KeywordCoverageEvaluator
from llm_eval_framework.evaluators.latency_check import LatencyCheckEvaluator
from llm_eval_framework.evaluators.toxicity_check import ToxicityCheckEvaluator
from llm_eval_framework.providers.mock_provider import MockProvider
from llm_eval_framework.providers.openai_provider import OpenAIProvider
from llm_eval_framework.reporting.report_generator import generate_reports
from llm_eval_framework.utils.config_loader import load_yaml_config


def build_provider(provider_name: str):
    if provider_name == "mock":
        return MockProvider()
    if provider_name == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unsupported provider: {provider_name}")


def parse_keywords(value: Any) -> List[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [keyword.strip() for keyword in str(value).split("|") if keyword.strip()]


def run_evaluation(dataset_path: str, config_path: str, provider_name: str) -> List[Dict[str, Any]]:
    config = load_yaml_config(config_path)
    evaluation_config = config["evaluation"]
    reporting_config = config["reporting"]

    provider = build_provider(provider_name)
    dataset = pd.read_csv(dataset_path)

    evaluators = [
        KeywordCoverageEvaluator(
            min_score=float(evaluation_config["min_keyword_score"]),
            case_sensitive=bool(evaluation_config["case_sensitive"]),
        ),
        ToxicityCheckEvaluator(),
        LatencyCheckEvaluator(max_latency_ms=int(evaluation_config["max_latency_ms"])),
    ]

    results: List[Dict[str, Any]] = []

    for _, row in dataset.iterrows():
        response = provider.generate(str(row["prompt"]))
        metadata = {
            "required_keywords": parse_keywords(row.get("required_keywords")),
            "latency_ms": response.latency_ms,
            "category": row.get("category"),
        }

        evaluation_results = [
            evaluator.evaluate(
                expected=str(row.get("expected_answer", "")),
                actual=response.text,
                metadata=metadata,
            )
            for evaluator in evaluators
        ]

        passed = all(result.passed for result in evaluation_results)
        results.append(
            {
                "id": row["id"],
                "category": row.get("category", ""),
                "prompt": row["prompt"],
                "expected_answer": row.get("expected_answer", ""),
                "actual_response": response.text,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "passed": passed,
                "evaluations": [result.__dict__ for result in evaluation_results],
            }
        )

    generate_reports(
        results=results,
        output_dir=reporting_config["output_dir"],
        json_name=reporting_config["json_report"],
        html_name=reporting_config["html_report"],
    )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LLM evaluation tests")
    parser.add_argument("--dataset", required=True, help="Path to evaluation dataset CSV")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--provider", default="mock", choices=["mock", "openai"], help="LLM provider")
    args = parser.parse_args()

    results = run_evaluation(args.dataset, args.config, args.provider)
    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    print(f"Evaluation completed: {passed}/{total} passed")

    if passed != total:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
