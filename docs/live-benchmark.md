# Live paired QE benchmark

Install the project and set OPENAI_API_KEY. Use a pinned dataset revision (for example AI-QE-Eval-500 commit 14962aaf13c088d1c40674967ec6781853548227), preferably its held-out test split. Choose two compatible OpenAI chat models and a distinct judge supporting JSON mode. Prices are operator-supplied USD per million tokens; verify current provider pricing.

```bash
python -m llm_eval_framework.live_benchmark --dataset test.jsonl --baseline-model BASE --candidate-model CANDIDATE --judge-model JUDGE --baseline-input-price 1 --baseline-output-price 1 --candidate-input-price 1 --candidate-output-price 1 --limit 5
```

The prices above are placeholders, not actual provider rates. Start with five cases; increase only after reviewing costs. Each case makes two generation and two judge calls, excluding retries. Reports contain raw answers, judgments, model IDs, prompts, dataset hash, per-category rubric scores, latency, tokens and generation-only estimated cost. No context means groundedness is unavailable. The first N cases are a smoke sample, not a representative benchmark. Full evaluation requires a representative held-out set, human judge calibration and repeated runs. No live results are claimed. Do not send confidential data.
