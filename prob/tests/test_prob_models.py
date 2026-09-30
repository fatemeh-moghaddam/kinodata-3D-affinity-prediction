"""prob_models: the probe registry prob_orchestrate runs, and the two custom
estimators behaving as sklearn estimators inside Pipeline/GridSearchCV."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from prob.prob_models import (
    LINEAR_PROBES,
    NONLINEAR_PROBES,
    HybridRandomForestRegressor,
    TorchMLPRegressor,
    _resolve_activation,
)
from prob.prob_run import _make_pipeline


@pytest.mark.parametrize("entry", LINEAR_PROBES + NONLINEAR_PROBES, ids=lambda e: e["name"])
def test_registry_entries_fit_the_pipeline(entry):
    assert set(entry) <= {"name", "estimator", "param_grid", "n_jobs"}
    pipe = _make_pipeline(clone(entry["estimator"]))
    valid = pipe.get_params()
    for key in entry["param_grid"]:
        assert key.startswith("model__"), key
        assert key in valid, f"{key} is not a parameter of {type(entry['estimator']).__name__}"


def test_registry_names_are_unique_and_path_safe():
    names = [e["name"] for e in LINEAR_PROBES + NONLINEAR_PROBES]
    assert len(names) == len(set(names))
    assert all("/" not in n and " " not in n for n in names)


def test_nonlinear_estimators_expose_device():
    """main() moves these onto prob_config.device through this attribute."""
    assert all(hasattr(e["estimator"], "device") for e in NONLINEAR_PROBES)


@pytest.fixture
def xy():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    return X, X[:, 0] * 2 - X[:, 1] + 0.05 * rng.normal(size=300)


def test_torch_mlp_learns_and_is_reproducible(xy):
    X, y = xy
    kw = dict(hidden_layer_sizes=(16,), max_iter=60, random_state=0, device="cpu")
    a = TorchMLPRegressor(**kw).fit(X, y).predict(X)
    b = TorchMLPRegressor(**kw).fit(X, y).predict(X)
    np.testing.assert_allclose(a, b)
    assert a.shape == (300,)
    assert np.corrcoef(a, y)[0, 1] > 0.9


def test_torch_mlp_rejects_unknown_activation():
    with pytest.raises(ValueError, match="activation"):
        _resolve_activation("gelu")


def test_hybrid_rf_cpu_backend_in_gridsearch(xy):
    X, y = xy
    search = GridSearchCV(
        _make_pipeline(HybridRandomForestRegressor(random_state=0)),
        {"model__n_estimators": [5], "model__max_depth": [None, 3]},
        cv=2,
    ).fit(X, y)
    assert type(search.best_estimator_["model"].backend_).__name__ == "RandomForestRegressor"
    assert search.predict(X).shape == (300,)
