"""Generate HTML reports summarising evaluation metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from jinja2 import Template

from src.common.fs import ensure_dir


TEMPLATE = Template(
    """<!DOCTYPE html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\" />
  <title>SynMedVision Evaluation Report</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 2rem; }
    h1 { color: #2b7a78; }
    table { border-collapse: collapse; margin-bottom: 2rem; }
    th, td { border: 1px solid #ccc; padding: 0.5rem 0.75rem; }
    th { background: #def2f1; }
  </style>
</head>
<body>
  <h1>SynMedVision Evaluation Report</h1>
  {% for section, values in metrics.items() %}
  <h2>{{ section|title }}</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {% for key, value in values.items() %}
    <tr><td>{{ key }}</td><td>{{ value }}</td></tr>
    {% endfor %}
  </table>
  {% endfor %}
</body>
</html>
"""
)


def generate_report(metrics: Dict[str, Dict[str, Any]], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    out_path.write_text(TEMPLATE.render(metrics=metrics), encoding="utf-8")
    return out_path


__all__ = ["generate_report"]
