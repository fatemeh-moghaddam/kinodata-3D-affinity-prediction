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

    for key in ("depthAxis", "colourBy", "facetRowBy", "facetColBy",
                "panelsPerRow", "yScale", "matrixRow", "matrixCol"):
        assert key in defaults, f"{key} missing from config.defaults"

    # a facet axis may legitimately be unset; the others must name real factors
    for key in ("depthAxis", "colourBy", "matrixRow", "matrixCol"):
        assert defaults[key] in FKEYS, f"{key} is not a factor"
    for key in ("facetRowBy", "facetColBy"):
        assert defaults[key] == "" or defaults[key] in FKEYS

    # the two facet axes must not be the same factor, or the grid is a diagonal
    if defaults["facetRowBy"] and defaults["facetColBy"]:
        assert defaults["facetRowBy"] != defaults["facetColBy"]
    # nor may either be the x axis
    assert defaults["depthAxis"] not in (defaults["facetRowBy"], defaults["facetColBy"])

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
