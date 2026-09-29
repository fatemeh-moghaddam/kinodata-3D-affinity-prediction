def pytest_configure(config):
    config.addinivalue_line(
        "markers", "folds: compares aggregated files against the per-fold files (needs fold dirs 0-4)"
    )
