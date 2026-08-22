"""Standalone, self-contained explorer page for the probing sweep.

    from prob.explorer import build_explorer
    build_explorer("explorer.html")

or from the shell:

    uv run python -m prob.explorer --open
"""
from prob.explorer.build import (
    FACTORS,
    METRICS,
    EXTRA_COLUMNS,
    Column,
    Factor,
    Metric,
    build_explorer,
    build_payload,
    collect_runs,
    drop_incomplete,
    render_html,
)

__all__ = [
    "FACTORS", "METRICS", "EXTRA_COLUMNS",
    "Factor", "Metric", "Column",
    "collect_runs", "drop_incomplete", "build_payload", "render_html", "build_explorer",
]
