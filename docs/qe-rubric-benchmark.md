# QE rubric benchmark

Use a pinned revision of [AI-QE-Eval-500](https://github.com/aditya42/AI-QE-Eval-500), preferably its held-out test JSONL. Generate responses using only the input field. An independent judge or human reviewer assigns each criterion a score from 0 to 4. Supply two JSON arrays of judgments, e.g. [{"id":"AIQE-0001","scores":{"gap_detection":3,"clarifying_questions":3,"testable_rewrite":4,"risk_awareness":3}}]. Include exactly one judgment per dataset record.

```bash
python -m llm_eval_framework.qe_rubric --dataset test.jsonl --baseline baseline.json --candidate candidate.json --output rubric-report.json
```

The gate requires candidate mean >=3, no pass-rate regression overall or by category, and no mean-score regression overall or by category beyond --max-drop (default zero). The report includes dataset hash and category, difficulty, risk and failure-category breakdowns. Exit 1 signals regression. This is a scorer of externally supplied judgments, not a live judge or proof of model quality. Retain model/prompt/judge versions and raw responses separately; adjudicate disagreements as described in the dataset evaluation guide.
