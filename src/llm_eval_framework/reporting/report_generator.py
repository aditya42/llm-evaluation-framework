import json
from pathlib import Path
from typing import Any, Dict, List
from jinja2 import Template


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>LLM Evaluation Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 32px; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; vertical-align: top; }
        th { background: #f4f4f4; }
        .pass { color: green; font-weight: bold; }
        .fail { color: red; font-weight: bold; }
    </style>
</head>
<body>
<h1>LLM Evaluation Report</h1>
<p>Total: {{ summary.total }} | Passed: {{ summary.passed }} | Failed: {{ summary.failed }} | Pass Rate: {{ summary.pass_rate }}%</p>
<table>
<tr><th>ID</th><th>Category</th><th>Prompt</th><th>Actual Response</th><th>Status</th><th>Scores</th></tr>
{% for row in results %}
<tr>
<td>{{ row.id }}</td>
<td>{{ row.category }}</td>
<td>{{ row.prompt }}</td>
<td>{{ row.actual_response }}</td>
<td class="{{ 'pass' if row.passed else 'fail' }}">{{ 'PASS' if row.passed else 'FAIL' }}</td>
<td><pre>{{ row.evaluations }}</pre></td>
</tr>
{% endfor %}
</table>
</body>
</html>
"""


def generate_reports(results: List[Dict[str, Any]], output_dir: str, json_name: str, html_name: str) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    failed = total - passed
    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": round((passed / total) * 100, 2) if total else 0,
    }

    json_payload = {"summary": summary, "results": results}
    (output_path / json_name).write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

    template = Template(HTML_TEMPLATE)
    html = template.render(summary=summary, results=results)
    (output_path / html_name).write_text(html, encoding="utf-8")
