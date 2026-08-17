from datetime import UTC, datetime
from html import escape
from pathlib import Path
from uuid import uuid4

from ..constants import EDA_REPORT_DIRECTORY


def create_html_report(
    project, overview: dict, profiles: list[dict], quality: dict, correlation: dict
) -> tuple[dict, Path]:
    report_id = uuid4().hex
    path = EDA_REPORT_DIRECTORY / f"{project.id}_{report_id}.html"
    profile_rows = "".join(
        f"<tr><td>{escape(str(item['name']))}</td><td>{escape(str(item['semantic_type']))}</td>"
        f"<td>{item['non_null_count']}</td><td>{item['null_count']}</td><td>{item['unique_count']}</td></tr>"
        for item in profiles[:200]
    )
    missing_rows = "".join(
        f"<tr><td>{escape(str(item['column']))}</td><td>{item['count']}</td><td>{item['percentage']}%</td></tr>"
        for item in quality["findings"]["missing_by_column"][:200]
    )
    missing_chart = (
        "".join(
            f'<div style="margin:8px 0"><span>{escape(str(item["column"]))} — {item["percentage"]}%</span>'
            f'<div style="height:12px;background:#e7ecf5;border-radius:6px"><div style="height:12px;width:{min(float(item["percentage"]), 100)}%;background:#3b82f6;border-radius:6px"></div></div></div>'
            for item in quality["findings"]["missing_by_column"][:50]
            if item["count"]
        )
        or "<p>No missing values detected.</p>"
    )
    numeric_rows = "".join(
        f"<tr><td>{escape(str(item['name']))}</td><td>{item.get('mean', '—')}</td><td>{item.get('median', '—')}</td><td>{item.get('standard_deviation', '—')}</td><td>{item.get('potential_outlier_count', 0)}</td></tr>"
        for item in profiles
        if item.get("semantic_type") == "numeric"
    )
    categorical_rows = "".join(
        f"<tr><td>{escape(str(item['name']))}</td><td>{escape(str(item.get('most_frequent_value', '—')))}</td><td>{item.get('most_frequent_count', 0)}</td><td>{item.get('unique_count', 0)}</td></tr>"
        for item in profiles
        if item.get("semantic_type") in {"categorical", "boolean", "text"}
    )
    correlation_rows = "".join(
        f"<tr><th>{escape(str(name))}</th>{''.join(f'<td>{value:.3f}</td>' if value is not None else '<td>—</td>' for value in row)}</tr>"
        for name, row in zip(
            correlation.get("columns", []), correlation.get("matrix", [])
        )
    )
    correlation_header = "".join(
        f"<th>{escape(str(name))}</th>" for name in correlation.get("columns", [])
    )
    generated = datetime.now(UTC)
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>EDA Report</title>
<style>body{{font:15px system-ui;color:#172033;max-width:1100px;margin:40px auto;padding:0 24px}}h1{{color:#2457d6}}.grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}}.card{{background:#f3f6fc;padding:18px;border-radius:10px}}table{{width:100%;border-collapse:collapse;margin:16px 0}}th,td{{padding:9px;border-bottom:1px solid #dde3ee;text-align:left}}small{{color:#667085}}</style></head><body>
<h1>EDA Hub Report</h1><p><strong>{escape(project.original_filename)}</strong></p><small>Generated {generated.isoformat()} · Analysis version {escape(project.analysis_version)}</small>
<h2>Overview</h2><div class="grid"><div class="card"><b>Rows</b><br>{project.rows:,}</div><div class="card"><b>Columns</b><br>{project.columns:,}</div><div class="card"><b>Missing cells</b><br>{overview['missing_values']:,} ({overview['missing_percentage']}%)</div><div class="card"><b>Duplicate rows</b><br>{overview['duplicate_rows']:,}</div></div>
<h2>Column profiles</h2><table><thead><tr><th>Column</th><th>Semantic type</th><th>Non-null</th><th>Null</th><th>Unique</th></tr></thead><tbody>{profile_rows}</tbody></table>
<h2>Missing-value analysis</h2><table><thead><tr><th>Column</th><th>Missing</th><th>Percentage</th></tr></thead><tbody>{missing_rows}</tbody></table>
<h3>Missing-value chart</h3>{missing_chart}
<h2>Numeric statistics and outliers</h2><table><thead><tr><th>Column</th><th>Mean</th><th>Median</th><th>Std. deviation</th><th>IQR outliers</th></tr></thead><tbody>{numeric_rows}</tbody></table>
<h2>Categorical summaries</h2><table><thead><tr><th>Column</th><th>Top value</th><th>Frequency</th><th>Unique</th></tr></thead><tbody>{categorical_rows}</tbody></table>
<h2>Pearson correlation</h2><table><thead><tr><th></th>{correlation_header}</tr></thead><tbody>{correlation_rows}</tbody></table>
<h2>Data-quality findings</h2><pre>{escape(str({key: value for key, value in quality['findings'].items() if key != 'missing_by_column'}))}</pre>
<p><small>Charts and statistics are bounded by EDA Hub server limits. No browser-only state was used to generate this report.</small></p></body></html>"""
    path.write_text(html, encoding="utf-8")
    metadata = {
        "id": report_id,
        "format": "html",
        "path": str(path),
        "created_at": generated,
    }
    return metadata, path
