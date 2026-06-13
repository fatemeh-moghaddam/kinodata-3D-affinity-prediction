import pytest


def test_validate_spec_rejects_bad_values():
    pytest.importorskip("prob.run_extraction")
    from prob.run_extraction import ProbingJobSpec, _validate_spec

    with pytest.raises(ValueError):
        _validate_spec(ProbingJobSpec(gnn_model_type="BAD"))

    with pytest.raises(ValueError):
        _validate_spec(ProbingJobSpec(split_type="BAD"))

    with pytest.raises(ValueError):
        _validate_spec(ProbingJobSpec(filter_rmsd_max_value=123))

    with pytest.raises(ValueError):
        _validate_spec(ProbingJobSpec(k_fold=0))


def test_validate_spec_accepts_supported_rmsd_values():
    pytest.importorskip("prob.run_extraction")
    from prob.run_extraction import ProbingJobSpec, _validate_spec

    for v in (2, 4, 6, 2.0, 4.0, 6.0, None):
        _validate_spec(ProbingJobSpec(filter_rmsd_max_value=v))

