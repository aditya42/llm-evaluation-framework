# Independent LLM judge

Set OPENAI_API_KEY and choose a compatible chat model supporting JSON mode. Provide a JSON array of responses with id, answer and optional context (trusted evidence text). Do not submit confidential data.

```bash
python -m llm_eval_framework.llm_judge --dataset test.jsonl --responses candidate.json --judge-model YOUR_MODEL --output candidate-judgments.json
```

Repeat for baseline and use qe_rubric to compare. Groundedness is the fraction of judge-extracted factual claims labeled supported; unsupported_claim_rate includes contradicted and unsupported claims. No context or no extracted claims yields null, not a passing score. Evidence substring validation does not prove entailment or claim completeness. Judge outputs can be biased or manipulated; calibrate against human ratings and use an independent model. The original QE dataset references are scoring guidance, not factual evidence. Live API execution requires credentials and is not validated by offline tests. Groundedness metrics are per case and are not yet integrated into the aggregate regression gate.
