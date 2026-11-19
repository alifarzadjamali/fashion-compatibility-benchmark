import numpy as np
import pytest

from repbench.utils.paths import data_path
from repbench.v2.logistic import V2_C_GRID, GridEdgeError, V2LogisticCompatibility
from repbench.v2.selection import select_models


def test_data_root_is_outside_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("REPBENCH_DATA_ROOT", str(tmp_path))
    assert data_path("iqon_audit", "source.zip") == tmp_path / "iqon_audit/source.zip"


def selection_rows():
    return [
        {
            "dataset": "polyvore_d_clean",
            "split": "valid",
            "metric": "cp_auc",
            "setting": "native",
            "model_key": k,
            "value": v,
        }
        for k, v in [("generic", 0.8), ("fashion", 0.7), ("cnn", 0.6)]
    ]


FAMILIES = {
    "generic": "generic_vision_language",
    "fashion": "fashion_retrieval",
    "cnn": "supervised_cnn",
}


def test_selection_uses_clean_validation_and_complete_roster():
    result = select_models(selection_rows(), dataset="polyvore_d_clean", families=FAMILIES)
    assert result["overall"] == "generic"
    assert result["fashion"] == "fashion"
    rows = selection_rows()
    rows[0]["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        select_models(rows, dataset="polyvore_d_clean", families=FAMILIES)
    with pytest.raises(ValueError, match="Incomplete"):
        select_models(selection_rows()[:2], dataset="polyvore_d_clean", families=FAMILIES)
    with pytest.raises(ValueError, match="clean"):
        select_models(selection_rows(), dataset="historical_polyvore_d", families=FAMILIES)


def test_edge_grid_fails_closed_on_uninformative_validation():
    x = np.zeros((12, 2))
    y = np.tile([0, 1], 6)
    model = V2LogisticCompatibility(max_extensions=1)
    with pytest.raises(GridEdgeError):
        model.fit(x, y, x, y)
    assert len(model.grid_extensions_) == 1
    assert min(model.c_grid) == pytest.approx(1e-7)
    assert V2_C_GRID == tuple(10.0**i for i in range(-6, 5))
