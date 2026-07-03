from pathlib import Path
from llm_eval_framework.runner import run_evaluation


def test_runner_generates_results(tmp_path, monkeypatch):
    project_root = Path(__file__).resolve().parents[1]
    results = run_evaluation(
        dataset_path=str(project_root / "data" / "sample_eval_dataset.csv"),
        config_path=str(project_root / "configs" / "eval_config.yaml"),
        provider_name="mock",
    )
    assert len(results) == 4
    assert all("passed" in result for result in results)
