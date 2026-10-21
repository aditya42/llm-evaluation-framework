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


def test_mock_provider_deterministic_results():
    # Assumes the mock provider is deterministic and returns stable results on fixed inputs
    project_root = Path(__file__).resolve().parents[1]
    params = dict(
        dataset_path=str(project_root / "data" / "sample_eval_dataset.csv"),
        config_path=str(project_root / "configs" / "eval_config.yaml"),
        provider_name="mock",
    )

    results1 = run_evaluation(**params)
    results2 = run_evaluation(**params)

    # results are lists of dicts; assert exact equality for each element
    assert len(results1) == len(results2)
    for r1, r2 in zip(results1, results2):
        assert r1 == r2
