"""Run paired live QE benchmarks with an independent OpenAI-compatible judge."""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import mean
from llm_eval_framework.llm_judge import judge
from llm_eval_framework.qe_rubric import compare

def generate(client, model, task, prompt):
    start = time.perf_counter()
    response = client.chat.completions.create(model=model, messages=[{"role":"system","content":prompt},{"role":"user","content":task}], temperature=0)
    answer = response.choices[0].message.content
    if not answer or response.usage is None:
        raise ValueError("Missing answer or token usage")
    return dict(answer=answer, model=response.model, latency_ms=(time.perf_counter()-start)*1000, input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)

def summarize(rows, input_price, output_price):
    if not rows or any(not math.isfinite(p) or p < 0 for p in (input_price,output_price)):
        raise ValueError("Invalid rows or prices")
    i=sum(r["input_tokens"] for r in rows)
    o=sum(r["output_tokens"] for r in rows)
    return dict(count=len(rows), mean_latency_ms=mean(r["latency_ms"] for r in rows), input_tokens=i, output_tokens=o, estimated_generation_cost_usd=(i*input_price+o*output_price)/1_000_000)

def main():
    from openai import OpenAI
    p=argparse.ArgumentParser()
    for name in ("dataset","baseline-model","candidate-model","judge-model"):
        p.add_argument("--"+name,required=True)
    for name in ("baseline-input-price","baseline-output-price","candidate-input-price","candidate-output-price"):
        p.add_argument("--"+name,type=float,required=True)
    p.add_argument("--limit",type=int,default=5)
    p.add_argument("--baseline-prompt",default="Answer the quality engineering task accurately.")
    p.add_argument("--candidate-prompt",default="Answer the quality engineering task accurately.")
    p.add_argument("--output",default="live-benchmark.json")
    a=p.parse_args()
    if a.limit<1 or a.limit>500 or a.judge_model in (a.baseline_model,a.candidate_model):
        p.error("Limit must be 1–500 and judge model must differ from both candidate models")
    prices=[a.baseline_input_price,a.baseline_output_price,a.candidate_input_price,a.candidate_output_price]
    if any(not math.isfinite(v) or v<0 for v in prices):
        p.error("Prices must be finite and nonnegative")
    raw=Path(a.dataset).read_bytes()
    records=[json.loads(line) for line in raw.decode().splitlines() if line.strip()][:a.limit]
    if not records or len({r["id"] for r in records})!=len(records):
        raise ValueError("Empty or duplicate dataset")
    client=OpenAI(timeout=60,max_retries=2)
    outputs={"baseline":[],"candidate":[]}
    judgments={"baseline":[],"candidate":[]}
    for r in records:
        for side in ("baseline","candidate"):
            row=generate(client,getattr(a,side+"_model"),r["input"],getattr(a,side+"_prompt"))
            row["id"]=r["id"]
            outputs[side].append(row)
            judgments[side].append(judge(client,a.judge_model,r,row["answer"],r.get("context","")))
    report=compare(records,judgments["baseline"],judgments["candidate"])
    report.update(dataset_sha256=hashlib.sha256(raw).hexdigest(),case_ids=[r["id"] for r in records],judge_model=a.judge_model,outputs=outputs,judgments=judgments,prompts={"baseline":a.baseline_prompt,"candidate":a.candidate_prompt})
    report["usage"]={side:summarize(outputs[side],getattr(a,side+"_input_price"),getattr(a,side+"_output_price")) for side in outputs}
    report["cost_scope"]="Generation only; excludes judge, retries and infrastructure. Prices supplied by operator in USD per million tokens."
    Path(a.output).write_text(json.dumps(report,indent=2))
    return 0 if report["passed"] else 1

if __name__=="__main__":
    raise SystemExit(main())
