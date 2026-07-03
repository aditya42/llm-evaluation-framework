# LLM Test Evaluation Framework

A Python-based framework for evaluating LLM responses across quality, safety, correctness, faithfulness, latency, and regression stability.

## Why this project matters

This project demonstrates practical LLM QA / AI Quality Engineering skills:

- Prompt-response test automation
- LLM-as-judge evaluation
- Rule-based quality checks
- Dataset-driven regression testing
- JSON/HTML reporting
- CI execution through GitHub Actions
- Extensible provider design for OpenAI, local models, or mock providers

## Architecture

```text
llm-test-evaluation-framework/
├── configs/
│   └── eval_config.yaml
├── data/
│   └── sample_eval_dataset.csv
├── reports/
├── src/
│   └── llm_eval_framework/
│       ├── evaluators/
│       │   ├── base.py
│       │   ├── exact_match.py
│       │   ├── keyword_coverage.py
│       │   ├── toxicity_check.py
│       │   └── latency_check.py
│       ├── providers/
│       │   ├── base.py
│       │   ├── mock_provider.py
│       │   └── openai_provider.py
│       ├── reporting/
│       │   └── report_generator.py
│       ├── runner.py
│       └── utils/
│           └── config_loader.py
├── tests/
├── .github/workflows/
│   └── ci.yml
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Run evaluation with mock provider

```bash
python -m llm_eval_framework.runner \
  --dataset data/sample_eval_dataset.csv \
  --config configs/eval_config.yaml \
  --provider mock
```

## Run tests

```bash
pytest -v
```

## Output

Reports are generated under `reports/`:

- `evaluation_report.json`
- `evaluation_report.html`

## Dataset format

```csv
id,prompt,expected_answer,required_keywords,category
TC001,"What is CI/CD?","Continuous integration and continuous delivery","continuous integration|continuous delivery",definition
```

## Next enhancements

- Add semantic similarity using sentence-transformers
- Add hallucination/faithfulness checks against retrieved context
- Add LLM judge scoring rubric
- Add dashboards with trend history
- Add support for batch model comparison
