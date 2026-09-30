"""prob_metrics and prob_stats: metrics, bootstrap CIs, pairing by ident, and
paired comparisons with Holm-Bonferroni correction."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from prob.prob_metrics import evaluate_predictions, pearson, regression_scorers
from prob.prob_stats import (
    _holm_bonferroni_correction,
    align_predictions_on_ident,
    bootstrap_ci,
    compare_multiple_conditions,
    compare_two_conditions,
    run_probe_statistical_tests,
)


@pytest.fixture
def yy():
    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    good = y + 0.3 * rng.normal(size=300)
    bad = y + 1.5 * rng.normal(size=300)
    return y, good, bad


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────


def test_perfect_predictions():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert evaluate_predictions(y, y) == {"r2": 1.0, "rmse": 0.0, "mae": 0.0, "pearson": pytest.approx(1.0)}


def test_pearson_is_nan_for_constant_input():
    """A bootstrap resample can be constant; that must be NaN, not a crash or 0."""
    assert np.isnan(pearson(np.ones(5), np.arange(5.0)))
    assert np.isnan(pearson(np.arange(5.0), np.ones(5)))


def test_scorers_are_what_gridsearch_refits_on():
    assert set(regression_scorers()) == {"r2", "pearson", "neg_rmse", "neg_mae"}


# ─────────────────────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────────────────────


def test_bootstrap_ci_brackets_the_point_estimate_and_is_seeded(yy):
    y, good, _ = yy
    a = bootstrap_ci(y, good, n_bootstrap=200, random_state=1, name="r2")
    b = bootstrap_ci(y, good, n_bootstrap=200, random_state=1, name="r2")
    assert a == b
    assert a["metric"] == "r2"
    assert a["lower"] < a["point_estimate"] < a["upper"]


def test_statistical_tests_cover_every_metric(yy):
    y, good, _ = yy
    tests = run_probe_statistical_tests(y, good, n_bootstrap=50, random_state=0)
    assert set(tests) == {"r2_ci", "rmse_ci", "mae_ci", "pearson_ci"}
    assert tests["rmse_ci"]["point_estimate"] == pytest.approx(evaluate_predictions(y, good)["rmse"])


# ─────────────────────────────────────────────────────────────
# Pairing by ident
# ─────────────────────────────────────────────────────────────


def test_align_keeps_shared_idents_sorted():
    a = pd.DataFrame({"ident": [3, 1, 2], "y_true": [30.0, 10.0, 20.0], "y_pred": [3.0, 1.0, 2.0]})
    b = pd.DataFrame({"ident": [2, 4, 3], "y_true": [20.0, 40.0, 30.0], "y_pred": [-2.0, -4.0, -3.0]})
    out = align_predictions_on_ident({"a": a, "b": b})
    np.testing.assert_array_equal(out["a"][0], [20.0, 30.0])
    np.testing.assert_array_equal(out["a"][1], [2.0, 3.0])
    np.testing.assert_array_equal(out["b"][1], [-2.0, -3.0])


def test_align_rejects_different_targets():
    a = pd.DataFrame({"ident": [1, 2], "y_true": [1.0, 2.0], "y_pred": [0.0, 0.0]})
    b = pd.DataFrame({"ident": [1, 2], "y_true": [1.0, 9.0], "y_pred": [0.0, 0.0]})
    with pytest.raises(ValueError, match="y_true differs"):
        align_predictions_on_ident({"a": a, "b": b})


def test_align_rejects_old_csv_without_ident_and_disjoint_runs():
    no_ident = pd.DataFrame({"y_true": [1.0], "y_pred": [1.0]})
    with pytest.raises(ValueError, match="no ident column"):
        align_predictions_on_ident({"old": no_ident})

    a = pd.DataFrame({"ident": [1], "y_true": [1.0], "y_pred": [1.0]})
    b = pd.DataFrame({"ident": [2], "y_true": [1.0], "y_pred": [1.0]})
    with pytest.raises(ValueError, match="share no test idents"):
        align_predictions_on_ident({"a": a, "b": b})


# ─────────────────────────────────────────────────────────────
# Paired comparisons
# ─────────────────────────────────────────────────────────────


def test_better_condition_has_positive_r2_delta(yy):
    y, good, bad = yy
    res = compare_two_conditions("good", y, good, "bad", y, bad, n_bootstrap=200, random_state=0)
    assert res["delta_r2"] > 0 and res["delta_rmse"] < 0
    assert res["delta_r2_ci_low"] > 0
    assert res["delta_r2_bootstrap_pval"] < 0.05
    assert res["wilcoxon_pval"] < 0.05
    assert res["n_samples"] == len(y)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_identical_conditions_have_zero_deltas(yy):
    y, good, _ = yy
    res = compare_two_conditions("a", y, good, "b", y, good, n_bootstrap=50, random_state=0)
    for name in ("r2", "rmse", "mae", "pearson"):
        assert res[f"delta_{name}"] == 0.0
        assert res[f"delta_{name}_bootstrap_pval"] == 1.0


@pytest.mark.xfail(strict=True, reason=(
    "compare_two_conditions falls back to p=1.0 on ValueError, but scipy>=1.15 "
    "returns NaN (with a RuntimeWarning) when every paired difference is zero"
))
@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_identical_conditions_wilcoxon_pval_is_one(yy):
    y, good, _ = yy
    res = compare_two_conditions("a", y, good, "b", y, good, n_bootstrap=10, random_state=0)
    assert res["wilcoxon_pval"] == 1.0


def test_unpaired_conditions_are_rejected(yy):
    y, good, bad = yy
    with pytest.raises(ValueError, match="paired"):
        compare_two_conditions("a", y, good, "b", y[::-1], bad, n_bootstrap=10)


def test_holm_bonferroni_known_values():
    adjusted = _holm_bonferroni_correction(np.array([0.01, 0.04, 0.03]))
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])
    assert _holm_bonferroni_correction(np.array([0.5, 0.9])).max() <= 1.0
    assert len(_holm_bonferroni_correction(np.array([]))) == 0


def test_compare_multiple_conditions_all_pairs_corrected(yy):
    y, good, bad = yy
    df = compare_multiple_conditions(
        {"good": (y, good), "bad": (y, bad), "worse": (y, bad * 2)}, n_bootstrap=50, random_state=0,
    )
    assert len(df) == 3
    for col in ("delta_r2_bootstrap_pval", "wilcoxon_pval"):
        assert (df[f"{col}_holm"] >= df[col]).all()

    with pytest.raises(ValueError, match="at least 2"):
        compare_multiple_conditions({"only": (y, good)})
    with pytest.raises(ValueError, match="y_true mismatch"):
        compare_multiple_conditions({"a": (y, good), "b": (y[::-1], bad)})
