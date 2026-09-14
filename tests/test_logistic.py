from pathlib import Path

import yaml

from repbench.models.logistic import SHARED_C_GRID, LogisticCompatibility


def test_default_logistic_grid_is_shared_protocol_grid():
    assert SHARED_C_GRID == (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)
    assert LogisticCompatibility().c_grid == SHARED_C_GRID


def test_pilot_config_matches_shared_logistic_grid():
    config_path = Path(__file__).parents[1] / "configs" / "experiments" / "pilot.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert tuple(config["c_grid"]) == SHARED_C_GRID
