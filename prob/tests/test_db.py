"""
prob.db: ingest the probe-run files into Parquet, query them back through the
DuckDB views, and flag every file that cannot be taken at face value.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prob.db import connect, discover_sources, ingest, load_predictions, load_runs
from prob.db import ingestion as ingest_mod
from prob.db.ingestion import flatten_summary, read_predictions
from prob.db.schema import ISSUE_KINDS, TABLES

FACTORS = ("CGNN-3D", "rmsd_cutoff_2", "random-k-fold")


def write_run(root: Path, target: str, probe: str, layer: int | None, *,
              n: int = 6, ident: bool = True, summary: dict | str | None = "default",
              predictions: str | None = "default", best_params: dict | None = None,
              figures: tuple[str, ...] = ("parity",)) -> Path:
    """One run directory in the canonical (or, with layer=None, legacy) layout."""
    exp = root.joinpath(*FACTORS, target, probe)
    if layer is not None:
        exp = exp / str(layer)
    for sub in ("artifacts", "reports", "figures"):
        (exp / sub).mkdir(parents=True, exist_ok=True)

    if predictions == "default":
        rng = np.random.default_rng(layer or 0)
        frame = pd.DataFrame({"y_true": rng.normal(size=n), "y_pred": rng.normal(size=n)})
        if ident:
            frame.insert(0, "fold", np.arange(n) % 2)
            frame.insert(0, "ident", np.arange(100, 100 + n))
        frame.to_csv(exp / "artifacts" / f"{probe}_predictions.csv", index=False)
    elif predictions is not None:
        (exp / "artifacts" / f"{probe}_predictions.csv").write_text(predictions)

    if summary == "default":
        summary = {
            "model": probe,
            "metrics_on_unseen_data": {"r2": 0.5, "rmse": 1.0, "mae": 0.8, "spearman": 0.7},
            "n_samples": 10 * n, "n_train_samples": 9 * n, "n_test_samples": n, "n_features": 64,
            "statistical_tests": {
                "r2_ci": {"metric": "r2", "point_estimate": 0.5, "lower": 0.4, "upper": 0.6,
                          "confidence": 0.95},
                "pearson_ci": {"metric": "pearson", "point_estimate": 0.71, "lower": 0.6,
                               "upper": 0.8, "confidence": 0.95},
            },
            "probe_mode": "per_checkpoint",
            "n_splits_cv": 2,
            "across_checkpoints": {"r2": {"mean": 0.45, "sd": 0.05},
                                   "rmse": {"mean": 1.1, "sd": float("nan")}},
            "per_fold": [
                {"fold": k, "n_train_samples": 9, "n_test_samples": 3, "r2": 0.4 + k / 10,
                 "rmse": 1.0, "mae": 0.8, "pearson": 0.7, "best_params": {"model__alpha": 0.1},
                 "params_source": "tuned"}
                for k in range(2)
            ],
        }
    if isinstance(summary, dict):
        (exp / "reports" / f"{probe}_summary.json").write_text(json.dumps(summary))
    elif isinstance(summary, str):
        (exp / "reports" / f"{probe}_summary.json").write_text(summary)

    if best_params is not None:
        (exp / "reports" / f"{probe}_best_params.json").write_text(json.dumps(best_params))
    for kind in figures:
        (exp / "figures" / f"{probe}_{kind}.png").write_bytes(b"\x89PNG")
    return exp


@pytest.fixture
def tree(tmp_path):
    """data/probing with a clean run, its baseline, and one of each problem."""
    live = tmp_path / "probing"
    write_run(live, "affinity", "ridge", 0, best_params={"model__alpha": 1.0})
    write_run(live, "affinity", "ridge", 1)
    write_run(live, "affinity_shuffled_ident", "ridge", 0)
    write_run(live, "mw", "mlp", 0, predictions="y_true,y_pred\n1.0,2.0\n3x,\x00\x01\n")
    write_run(live, "mw", "mlp", 1, summary="")
    write_run(live, "mw", "ridge", 0, predictions=None)
    write_run(live, "mw", "lasso", 0, summary=None)
    write_run(live, "mw", "lasso", None)
    stray = live / "CGNN-3D" / "artifacts"          # predictions where no run can be
    stray.mkdir(parents=True)
    (stray / "x_predictions.csv").write_text("y_true,y_pred\n1,1\n")
    return tmp_path


def run_ingest(tmp_path: Path, **kwargs):
    return ingest(root=tmp_path / "probing", archive_root=tmp_path / "probing_archive",
                  db_dir=tmp_path / "probing_db", log=None, **kwargs)


# ─────────────────────────────────────────────────────────────
# Ingest
# ─────────────────────────────────────────────────────────────


def test_ingest_writes_every_table_per_source(tree):
    [res] = run_ingest(tree)
    assert res.status == "ingested"
    part = tree / "probing_db" / "source=current"
    assert sorted(p.stem for p in part.glob("*.parquet")) == sorted(t.name for t in TABLES)


def test_runs_carry_factors_metrics_and_open_columns(tree):
    run_ingest(tree)
    runs = load_runs(db_dir=tree / "probing_db", target="affinity", layer=0, prob_model="ridge",
                     include_baselines=False)
    assert len(runs) == 1
    r = runs.iloc[0]
    assert (r.gnn_model_type, r.rmsd_threshold, r.split_type, r.layer) == ("CGNN-3D", 2.0,
                                                                           "random-k-fold", 0)
    assert r.run_id == "CGNN-3D__rmsd2__random-k-fold__affinity__ridge__L0"
    assert (r.r2, r.r2_ci_lower, r.r2_ci_upper, r.ci_confidence) == (0.5, 0.4, 0.6, 0.95)
    assert r.pearson == pytest.approx(0.71)            # from the CI's point estimate
    assert r.spearman == pytest.approx(0.7)            # undeclared metric -> open column
    assert r.r2_ckpt_mean == 0.45 and r.r2_ckpt_lower == pytest.approx(0.40)
    assert np.isnan(r.rmse_ckpt_sd)                     # NaN in the JSON -> NULL
    assert r.probe_mode == "per_checkpoint" and r.n_test_samples == 6
    assert json.loads(r.best_params) == {"model__alpha": 1.0}
    assert json.loads(r.summary)["model"] == "ridge"
    assert r.has_predictions and r.has_ident and r.n_predictions == 6
    assert Path(r.predictions_path).is_file() and Path(r.summary_path).is_file()


def test_baselines_are_runs_and_can_be_excluded(tree):
    run_ingest(tree)
    db = tree / "probing_db"
    every = load_runs(db_dir=db, target="affinity")
    assert every.is_baseline.sum() == 1
    assert set(every.target_full) == {"affinity", "affinity_shuffled_ident"}
    assert not load_runs(db_dir=db, target="affinity", include_baselines=False).is_baseline.any()


def test_predictions_folds_and_figures(tree):
    run_ingest(tree)
    db = tree / "probing_db"
    rid = "CGNN-3D__rmsd2__random-k-fold__affinity__ridge__L0"
    preds = load_predictions(rid, source="current", db_dir=db)
    csv = pd.read_csv(load_runs(db_dir=db).set_index("run_id").loc[rid, "predictions_path"])
    np.testing.assert_allclose(preds.y_pred, csv.y_pred)
    assert preds.ident.tolist() == csv.ident.tolist() and preds.row.tolist() == list(range(6))

    con = connect(db)
    folds = con.execute("SELECT fold, r2, params_source, best_params FROM folds "
                        "WHERE run_id = ? ORDER BY fold", [rid]).fetchall()
    assert [(f, round(r2, 2), src) for f, r2, src, _ in folds] == [(0, 0.4, "tuned"),
                                                                   (1, 0.5, "tuned")]
    assert json.loads(folds[0][3]) == {"model__alpha": 0.1}
    fig = con.execute("SELECT kind, path FROM figures WHERE run_id = ?", [rid]).fetchone()
    assert fig[0] == "parity" and Path(fig[1]).is_file()


def test_old_predictions_without_ident_get_null_ident(tmp_path):
    write_run(tmp_path / "probing", "affinity", "ridge", 0, ident=False)
    run_ingest(tmp_path)
    preds = connect(tmp_path / "probing_db").sql("SELECT ident, fold FROM predictions").df()
    assert preds.ident.isna().all() and preds.fold.isna().all() and len(preds) == 6
    assert not load_runs(db_dir=tmp_path / "probing_db").has_ident.any()


# ─────────────────────────────────────────────────────────────
# Issues: nothing is dropped silently
# ─────────────────────────────────────────────────────────────


def test_every_problem_becomes_an_issue(tree):
    [res] = run_ingest(tree)
    issues = connect(tree / "probing_db").sql("SELECT kind, severity, run_id FROM issues").df()
    kinds = set(issues.kind)
    assert kinds == {"predictions_corrupt", "summary_unreadable", "predictions_missing",
                     "summary_missing", "legacy_layout", "unrecognised_path"}
    assert all(ISSUE_KINDS[k][0] == s for k, s in zip(issues.kind, issues.severity))
    assert res.n_errors == 2 and res.n_warnings == len(issues) - 2


def test_corrupt_predictions_keep_the_run_but_no_rows(tree):
    run_ingest(tree)
    db = tree / "probing_db"
    run = load_runs(db_dir=db, target="mw", prob_model="mlp", layer=0).iloc[0]
    assert not run.has_predictions and run.r2 == 0.5
    assert load_predictions(run.run_id, source="current", db_dir=db).empty


def test_read_predictions_names_the_bad_rows(tmp_path):
    bad = tmp_path / "p.csv"
    bad.write_text("y_true,y_pred\n1.0,2.0\n3x,4.0\n5.0,6.0\n")
    frame, err = read_predictions(bad)
    assert frame is None and err.startswith("1 of 3 rows") and "line 3" in err
    bad.write_text("y_true,y_pred\n1.0,2.0,9\n")
    assert read_predictions(bad)[1].startswith("cannot parse")
    bad.write_text("a,b\n1,2\n")
    assert "missing column" in read_predictions(bad)[1]


def test_duplicate_factors_get_a_suffix(tmp_path):
    live = tmp_path / "probing"
    write_run(live, "affinity", "ridge", 0)
    # A hand-moved run: directory says "lasso", file says "ridge" -> same factors.
    exp = write_run(live, "affinity", "lasso", 0)
    for sub, suffix in (("artifacts", "_predictions.csv"), ("reports", "_summary.json")):
        (exp / sub / f"lasso{suffix}").rename(exp / sub / f"ridge{suffix}")
    run_ingest(tmp_path)
    con = connect(tmp_path / "probing_db")
    ids = sorted(r[0] for r in con.sql("SELECT run_id FROM runs").fetchall())
    assert ids[1] == ids[0] + "-2"
    assert con.sql("SELECT kind FROM issues").fetchall() == [("duplicate_run",)]


# ─────────────────────────────────────────────────────────────
# Re-ingesting
# ─────────────────────────────────────────────────────────────


def test_unchanged_sources_are_skipped_and_changes_rebuild(tree):
    run_ingest(tree)
    assert run_ingest(tree)[0].status == "unchanged"

    csv = next((tree / "probing").rglob("affinity/ridge/1/artifacts/*.csv"))
    st = csv.stat()
    os.utime(csv, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    assert run_ingest(tree)[0].status == "ingested"
    assert run_ingest(tree, force=True)[0].status == "ingested"


def test_schema_version_bump_rebuilds(tree, monkeypatch):
    run_ingest(tree)
    monkeypatch.setattr(ingest_mod, "SCHEMA_VERSION", ingest_mod.SCHEMA_VERSION + 1)
    assert run_ingest(tree)[0].status == "ingested"


def test_new_runs_show_up_after_ingest(tree):
    run_ingest(tree)
    db = tree / "probing_db"
    before = len(load_runs(db_dir=db))
    write_run(tree / "probing", "affinity", "ridge", 2)
    run_ingest(tree)
    assert len(load_runs(db_dir=db)) == before + 1


# ─────────────────────────────────────────────────────────────
# Several sources
# ─────────────────────────────────────────────────────────────


def test_archives_are_separate_sources_with_their_own_columns(tree):
    snap = tree / "probing_archive" / "20260101_120000"
    write_run(snap, "affinity", "ridge", 0, ident=False,
              summary={"metrics_on_unseen_data": {"r2": 0.1}, "n_test_samples": 6})
    (tree / "probing_archive" / "_old_data").mkdir()          # not a snapshot

    names = [s.name for s in discover_sources(tree / "probing",
                                              archive_root=tree / "probing_archive")]
    assert names == ["current", "20260101_120000"]

    run_ingest(tree)
    db = tree / "probing_db"
    con = connect(db)
    labels = dict(con.sql("SELECT source, label FROM sources").fetchall())
    assert labels["20260101_120000"] == "2026-01-01 12:00:00"
    old = load_runs(db_dir=db, source="20260101_120000")
    assert len(old) == 1 and old.r2.iloc[0] == 0.1
    assert old.spearman.isna().all()                 # column only the newer source has
    assert len(load_runs(db_dir=db)) == len(load_runs(db_dir=db, source="current")) + 1


def test_stale_sources_are_kept_unless_pruned(tree):
    snap = tree / "probing_archive" / "20260101_120000"
    write_run(snap, "affinity", "ridge", 0)
    run_ingest(tree)
    for p in sorted(snap.rglob("*"), reverse=True):
        p.unlink() if p.is_file() else p.rmdir()
    snap.rmdir()

    run_ingest(tree)
    assert (tree / "probing_db" / "source=20260101_120000").is_dir()
    run_ingest(tree, prune=True)
    assert not (tree / "probing_db" / "source=20260101_120000").exists()


def test_connect_on_an_empty_database_has_typed_empty_views(tmp_path):
    con = connect(tmp_path / "nothing_here")
    for name in ("runs", "predictions", "folds", "figures", "issues", "sources"):
        assert con.sql(f"SELECT count(*) FROM {name}").fetchone() == (0,)
    assert load_runs(db_dir=tmp_path / "nothing_here").empty


def test_unknown_filters_and_sources_are_errors(tree):
    run_ingest(tree)
    with pytest.raises(TypeError):
        load_runs(db_dir=tree / "probing_db", targett="affinity")
    with pytest.raises(ValueError):
        run_ingest(tree, names=["no_such_source"])


def test_flatten_summary_ignores_non_numbers():
    flat = flatten_summary({"metrics_on_unseen_data": {"r2": 0.3, "note": "x", "ok": True},
                            "statistical_tests": {"r2_ci": "skipped"}})
    assert flat == {"r2": 0.3}
