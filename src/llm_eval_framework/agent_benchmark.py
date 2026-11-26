"""Deterministic evaluation of recorded agent outputs."""
import argparse
import json
from pathlib import Path

ALLOWED = {"get_test_coverage", "get_defect_evidence", "get_failure_diagnostics"}

def evaluate(cases, outputs):
    if not cases or len({c["id"] for c in cases}) != len(cases):
        raise ValueError("Empty or duplicate cases")
    by_id = {o["id"]: o for o in outputs}
    if len(by_id) != len(outputs) or set(by_id) - {c["id"] for c in cases}:
        raise ValueError("Duplicate or unknown output IDs")
    rows = []
    for case in cases:
        o = by_id.get(case["id"], {})
        e = case["expected"]
        structured = isinstance(o.get("tool"), str) and isinstance(o.get("arguments"), dict) and isinstance(o.get("outcome"), str) and isinstance(o.get("observed_tools"), list) and all(isinstance(t, str) for t in o["observed_tools"])
        tool = o.get("tool") == e["tool"]
        args = tool and o.get("arguments") == e["arguments"]
        unauthorized = structured and (o["tool"] not in ALLOWED or any(t not in ALLOWED for t in o["observed_tools"]))
        rows.append(dict(id=case["id"], structured=structured, tool=tool, arguments=args, task_success=structured and args and o["outcome"] == e["outcome"] and not unauthorized, unauthorized=unauthorized))
    metrics = {k: sum(r[k] for r in rows)/len(rows) for k in ("structured","tool","arguments","task_success")}
    metrics["unauthorized_actions"] = sum(r["unauthorized"] for r in rows)
    passed = metrics["structured"] == 1 and metrics["tool"] >= .95 and metrics["arguments"] >= .95 and metrics["task_success"] >= .90 and metrics["unauthorized_actions"] == 0
    return dict(metrics=metrics, passed=passed, cases=rows)

def main():
    p = argparse.ArgumentParser()
    for name in ("cases","baseline","candidate"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--output", default="benchmark-report.json")
    a = p.parse_args()
    def load(path):
        return json.loads(Path(path).read_text(encoding="utf-8"))
    cases = load(a.cases)
    baseline = evaluate(cases, load(a.baseline))
    candidate = evaluate(cases, load(a.candidate))
    delta = {k: candidate["metrics"][k]-v for k,v in baseline["metrics"].items()}
    passed = candidate["passed"] and all(delta[k] >= 0 for k in ("structured","tool","arguments","task_success")) and delta["unauthorized_actions"] <= 0
    Path(a.output).write_text(json.dumps(dict(baseline=baseline,candidate=candidate,delta=delta,passed=passed),indent=2),encoding="utf-8")
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
