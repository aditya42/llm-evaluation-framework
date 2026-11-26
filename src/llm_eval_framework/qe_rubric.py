"""Score externally judged AI-QE-Eval-500 responses; compare paired runs."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

def score(record, scores):
    rubric = record["rubric"]
    criteria = rubric["criteria"]
    names = [c["name"] for c in criteria]
    if not names or len(set(names)) != len(names) or set(scores) != set(names):
        raise ValueError("Missing, duplicate or unexpected rubric criteria")
    weights = [c["weight"] for c in criteria]
    if any(not math.isfinite(w) or w <= 0 for w in weights) or not math.isclose(sum(weights), 1):
        raise ValueError("Invalid rubric weights")
    if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 4 for v in scores.values()):
        raise ValueError("Scores must be finite numbers in [0,4]")
    threshold = rubric["passing_score"]
    if not 0 <= threshold <= 4:
        raise ValueError("Invalid passing threshold")
    return sum(c["weight"] * scores[c["name"]] for c in criteria)

def evaluate(records, judgments):
    ids = {r["id"] for r in records}
    by_id = {j["id"]: j for j in judgments}
    if not records or len(ids) != len(records) or len(by_id) != len(judgments) or set(by_id) != ids:
        raise ValueError("Benchmark and judgment IDs must match exactly")
    rows = []
    groups = defaultdict(list)
    for r in records:
        j = by_id[r["id"]]
        value = score(r, j["scores"])
        row = dict(id=r["id"], score=value, passed=value >= r["rubric"]["passing_score"])
        rows.append(row)
        for field in ("category", "difficulty", "risk", "failure_category"):
            groups[field + ":" + r[field]].append(row)
    def summarize(items):
        return dict(count=len(items), mean_score=mean(x["score"] for x in items), pass_rate=mean(x["passed"] for x in items))
    return dict(overall=summarize(rows), groups={k:summarize(v) for k,v in sorted(groups.items())}, cases=rows)

def compare(records, baseline, candidate, max_drop=0.0):
    if not math.isfinite(max_drop) or not 0 <= max_drop <= 4:
        raise ValueError("Invalid score drop")
    b, c = evaluate(records, baseline), evaluate(records, candidate)
    regressions = [k for k in b["groups"] if k.startswith("category:") and (b["groups"][k]["mean_score"]-c["groups"][k]["mean_score"] > max_drop or c["groups"][k]["pass_rate"] < b["groups"][k]["pass_rate"])]
    passed = c["overall"]["mean_score"] >= 3 and not regressions and c["overall"]["pass_rate"] >= b["overall"]["pass_rate"] and b["overall"]["mean_score"]-c["overall"]["mean_score"] <= max_drop
    return dict(baseline=b, candidate=c, category_regressions=regressions, passed=passed)

def main():
    p = argparse.ArgumentParser()
    for name in ("dataset","baseline","candidate"):
        p.add_argument("--"+name, required=True)
    p.add_argument("--output", default="rubric-report.json")
    p.add_argument("--max-drop", type=float, default=0)
    a = p.parse_args()
    raw = Path(a.dataset).read_bytes()
    records = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    report = compare(records, json.loads(Path(a.baseline).read_text()), json.loads(Path(a.candidate).read_text()), a.max_drop)
    report["dataset_sha256"] = hashlib.sha256(raw).hexdigest()
    Path(a.output).write_text(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
