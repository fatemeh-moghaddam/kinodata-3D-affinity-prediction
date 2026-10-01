"""Standalone, self-contained explorer page for the probing sweep.

    from prob.explorer import build_explorer
    build_explorer("explorer.html")

or from the shell:

    uv run python -m prob.explorer --open
"""
from prob.explorer.build import (
    BACKENDS,
    FACTORS,
    METRICS,
    EXTRA_COLUMNS,
    Column,
    Factor,
    Metric,
    Source,
    build_backends_payload,
    build_explorer,
    build_payload,
    build_sources_payload,
    collect_backends,
    collect_runs,
    collect_sources,
    drop_incomplete,
    render_html,
    write_parity_sidecars,
)

__all__ = [
    "BACKENDS", "FACTORS", "METRICS", "EXTRA_COLUMNS",
    "Factor", "Metric", "Column", "Source",
    "collect_runs", "collect_sources", "collect_backends", "drop_incomplete",
    "build_payload", "build_sources_payload", "build_backends_payload", "write_parity_sidecars",
    "render_html", "build_explorer",
]
