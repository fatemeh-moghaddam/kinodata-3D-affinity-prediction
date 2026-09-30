"""The explorer page is only as good as its payload: these pin that contract.

Nothing here touches the real sweep -- the frames are built by hand -- so the
tests stay fast and keep passing on a machine with no data/probing directory.
"""
import json
import re

import numpy as np
import pandas as pd
import pytest

from prob.explorer.build import (
    EXTRA_COLUMNS,
    FACTORS,
    METRICS,
    PLACEHOLDER,
    build_payload,
    drop_incomplete,
    render_html,
)

FKEYS = [f.key for f in FACTORS]


def make_runs(n=4, **overrides):
    """A minimal frame with every column the payload builder reads."""
    frame = pd.DataFrame({
        "target": ["affinity"] * n,
        "gnn_model_type": ["CGNN-3D"] * n,
        "layer": list(range(n)),
        "prob_model": ["ridge"] * n,
        "rmsd_threshold": [2.0] * n,
        "split_type": ["random-k-fold"] * n,
        "r2": np.linspace(0.1, 0.4, n),
        "rmse": np.linspace(1.2, 1.0, n),
        "mae": np.linspace(0.9, 0.8, n),
        "r2_ci_lower": np.linspace(0.08, 0.38, n),
        "r2_ci_upper": np.linspace(0.12, 0.42, n),
        "rmse_ci_lower": np.linspace(1.1, 0.9, n),
        "rmse_ci_upper": np.linspace(1.3, 1.1, n),
        "r2_baseline": [0.0] * n,
        "r2_delta": np.linspace(0.1, 0.4, n),
        "n_test_samples": [4124] * n,
        "n_features": [256] * n,
    })
    # Anything else the payload reads (metrics added later, their bands, extra
    # table columns) gets a number too, so the only missing columns in a test are
    # the ones it dropped on purpose.
    read = ([m.key for m in METRICS]
            + [k for m in METRICS for k in (m.ci_lower, m.ci_upper) if k]
            + [c.key for c in EXTRA_COLUMNS])
    for key in read:
        if key not in frame:
            frame[key] = np.linspace(0.1, 0.4, n)
    return frame.assign(**overrides)


def test_payload_carries_every_factor_and_metric():
    payload = build_payload(make_runs())

    assert [f["key"] for f in payload["config"]["factors"]] == FKEYS
    assert set(payload["config"]["metrics"]) == {m.key for m in METRICS}
    assert payload["config"]["metricOrder"] == [m.key for m in METRICS]

    for row in payload["runs"]:
        for key in FKEYS:
            assert key in row, f"{key} missing from a serialised run"
        for metric in METRICS:
            assert metric.key in row


def test_metric_spec_reaches_the_page_verbatim():
    """The page reads direction and comparability off the payload, not its own
    hard-coded table -- so a change in build.py has to show up here."""
    metrics = build_payload(make_runs())["config"]["metrics"]

    assert metrics["r2"]["higherBetter"] is True
    assert metrics["rmse"]["higherBetter"] is False
    assert metrics["rmse"]["crossTarget"] is False
    assert metrics["r2"]["ciLower"] == "r2_ci_lower"
    assert metrics["r2"]["baseline"] == "r2_baseline"
    assert metrics["mae"]["ciLower"] is None


def test_default_view_names_both_facet_axes():
    """The depth-curve grid reads its row/column factors, wrap count and y-scale
    off the payload. A default naming something that is not a factor has to be
    survivable -- the page falls back rather than rendering nothing."""
    defaults = build_payload(make_runs())["config"]["defaults"]

    for key in ("depthAxis", "colourBy", "shapeBy", "facetRowBy", "facetColBy",
                "panelsPerRow", "yScale", "matrixRow", "matrixCol"):
        assert key in defaults, f"{key} missing from config.defaults"

    # The matrix needs both its axes -- a heatmap with one is a list -- so ""
    # is only meaningful for the depth curves' optional channels.
    for key in ("depthAxis", "colourBy", "matrixRow", "matrixCol"):
        assert defaults[key] in FKEYS, f"{key} is not a factor"
    for key in ("shapeBy", "facetRowBy", "facetColBy"):
        assert defaults[key] == "" or defaults[key] in FKEYS

    # the two facet axes must not be the same factor, or the grid is a diagonal
    if defaults["facetRowBy"] and defaults["facetColBy"]:
        assert defaults["facetRowBy"] != defaults["facetColBy"]
    # nor may either be the x axis
    assert defaults["depthAxis"] not in (defaults["facetRowBy"], defaults["facetColBy"],
                                        defaults["shapeBy"])

    assert defaults["panelsPerRow"] == "auto" or 1 <= int(defaults["panelsPerRow"]) <= 4
    assert defaults["yScale"] in ("free", "row", "shared")


def test_page_never_opens_on_a_silent_average():
    """Runs colliding on one mark must not be averaged unless that was asked
    for. The opening mode has to be one of the two that shows every run."""
    defaults = build_payload(make_runs())["config"]["defaults"]

    assert "aggregate" in defaults
    assert defaults["aggregate"] in ("split", "mean", "median", "break")
    assert defaults["aggregate"] in ("split", "break"), (
        "the page would open silently averaging collided runs")


def test_nan_becomes_null_not_the_string_nan():
    """json.dumps writes bare NaN, which is not valid JSON and makes the page
    fail to parse -- every missing number has to arrive as null."""
    runs = make_runs()
    runs.loc[0, "r2_ci_lower"] = np.nan
    runs.loc[1, "r2_baseline"] = np.nan

    payload = build_payload(runs)
    assert payload["runs"][0]["r2_ci_lower"] is None
    assert payload["runs"][1]["r2_baseline"] is None

    blob = json.dumps(payload, allow_nan=False)   # raises if any NaN slipped in
    assert "NaN" not in blob


def test_missing_columns_are_filled_and_reported():
    runs = make_runs().drop(columns=["mae", "n_features"])
    payload = build_payload(runs)

    assert set(payload["config"]["missingColumns"]) == {"mae", "n_features"}
    assert payload["runs"][0]["mae"] is None


def test_integer_factors_stay_integers():
    """Layer arrives as a pandas nullable Int64 and must not reach the page as
    3.0 -- the level lookups compare it by string."""
    runs = make_runs().astype({"layer": "Int64"})
    payload = build_payload(runs)

    layers = [r["layer"] for r in payload["runs"]]
    assert all(isinstance(v, int) for v in layers), layers


def test_drop_incomplete_splits_on_any_missing_factor():
    runs = make_runs(n=4)
    runs["layer"] = pd.array([0, 1, None, 3], dtype="Int64")

    usable, dropped = drop_incomplete(runs)

    assert len(usable) == 3
    assert len(dropped) == 1
    assert usable["layer"].notna().all()


def test_render_html_escapes_closing_script_tags():
    """A target named like markup would otherwise close the <script> block early
    and break every view on the page."""
    runs = make_runs(n=1, target=["</script><b>x"])
    html = render_html(build_payload(runs))

    assert PLACEHOLDER not in html
    assert "</script><b>x" not in html
    assert "<\\/script>" in html
    # exactly the script tags the template itself declares
    assert html.count("</script>") == html.count("<script")


def test_render_html_rejects_a_template_without_the_placeholder(tmp_path):
    bare = tmp_path / "bare.html"
    bare.write_text("<p>no placeholder here</p>", encoding="utf-8")

    with pytest.raises(ValueError, match=PLACEHOLDER):
        render_html(build_payload(make_runs()), template=bare)


def test_built_page_is_self_contained_apart_from_google_fonts():
    """The page has to work from a file:// URL with no network -- the only
    remote references allowed are the Google Fonts stylesheet and its faces."""
    html = render_html(build_payload(make_runs()))

    remote = re.findall(r'(?:href|src)="(https?://[^"]+)"', html)
    assert all(u.startswith(("https://fonts.googleapis.com",
                            "https://fonts.gstatic.com")) for u in remote), remote


# ─────────────────────────────────────────────────────────────
# Sources (live sweep + archives) and parity sidecars
# ─────────────────────────────────────────────────────────────

from prob.explorer.build import (  # noqa: E402
    PARITY_CALLBACK,
    Source,
    _archive_label,
    build_sources_payload,
    list_archives,
    run_id,
    write_parity_sidecars,
)


def test_sources_payload_shares_config_and_opens_on_a_source_with_runs(tmp_path):
    """Right after archiving, data/probing is empty: the page must open on the
    archive rather than on a blank view."""
    empty = Source("current", "Current", "current", tmp_path, pd.DataFrame())
    arch = Source("20260929_160507", "2026-09-29 16:05:07", "archive", tmp_path, make_runs())
    payload = build_sources_payload([empty, arch], parity_dir="x_parity")

    assert payload["defaultSource"] == "20260929_160507"
    assert [s["key"] for s in payload["sources"]] == ["current", "20260929_160507"]
    assert payload["sources"][0]["nRuns"] == 0
    assert payload["sources"][1]["nRuns"] == 4
    assert payload["config"]["parityDir"] == "x_parity"
    # per-source provenance lives on the source, not the shared config
    assert "nRuns" not in payload["config"] and "source" not in payload["config"]
    json.dumps(payload, allow_nan=False)


def test_sources_payload_renders_into_the_template(tmp_path):
    src = Source("current", "Current", "current", tmp_path, make_runs())
    html = render_html(build_sources_payload([src]))
    assert PLACEHOLDER not in html
    assert '"sources"' in html


def test_archives_list_newest_first_and_get_readable_labels(tmp_path):
    for name in ("20260901_120000", "20260929_160507", "20260915_080000"):
        (tmp_path / name).mkdir()
    (tmp_path / "stray.txt").write_text("not a snapshot")

    assert [p.name for p in list_archives(tmp_path)] == [
        "20260929_160507", "20260915_080000", "20260901_120000"]
    assert _archive_label("20260929_160507") == "2026-09-29 16:05:07"
    assert _archive_label("hand-made") == "hand-made"
    assert list_archives(tmp_path / "missing") == []


def test_run_id_is_file_safe_and_distinguishes_every_factor():
    row = make_runs(n=1).iloc[0]
    rid = run_id(row)
    assert re.fullmatch(r"[A-Za-z0-9_.-]+", rid), rid
    assert "rmsd2" in rid and "L0" in rid          # 2.0 -> "2", not "2.0"
    for key, other in (("layer", 3), ("target", "mw"), ("prob_model", "mlp"),
                       ("gnn_model_type", "DTI"), ("rmsd_threshold", 4.0),
                       ("split_type", "scaffold-k-fold")):
        changed = row.copy()
        changed[key] = other
        assert run_id(changed) != rid, key


def test_parity_sidecars_are_written_cached_and_pruned(tmp_path):
    csv = tmp_path / "ridge_predictions.csv"
    pd.DataFrame({"y_true": [1.0, 2.0, np.nan, 4.0],
                  "y_pred": [1.1, 1.9, 3.0, 3.5]}).to_csv(csv, index=False)
    runs = make_runs(n=2).assign(predictions_path=[csv, tmp_path / "gone.csv"])
    out = tmp_path / "side"
    (out).mkdir()
    (out / "stale.js").write_text("old")

    ids = write_parity_sidecars(runs, out, "current")

    assert ids.iloc[1] is None                      # CSV missing -> no sidecar
    js = (out / f"{ids.iloc[0]}.js").read_text()
    assert not (out / "stale.js").exists()
    key, body = re.fullmatch(
        rf"window\.{PARITY_CALLBACK}\((\"[^\"]+\"),(\{{.*\}})\);\n", js).groups()
    assert json.loads(key) == f"current/{ids.iloc[0]}"
    data = json.loads(body)
    assert data == {"t": [1, 2, 4], "p": [1.1, 1.9, 3.5]}   # NaN row dropped

    # unchanged CSV -> the sidecar is not rewritten
    before = (out / f"{ids.iloc[0]}.js").stat().st_mtime_ns
    write_parity_sidecars(runs, out, "current")
    assert (out / f"{ids.iloc[0]}.js").stat().st_mtime_ns == before


def test_payload_carries_the_parity_id_only_when_present():
    assert "parity_id" not in build_payload(make_runs())["runs"][0]
    runs = make_runs(n=2).assign(parity_id=["a", None])
    records = build_payload(runs)["runs"]
    assert records[0]["parity_id"] == "a" and records[1]["parity_id"] is None
