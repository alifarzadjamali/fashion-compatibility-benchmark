from types import SimpleNamespace

import pytest

from repbench.v2.state import remaining_seconds, require_budget, require_gate, sha256, write_json


def test_gate_rejects_stale_inputs(tmp_path):
    source = tmp_path / "source.py"
    source.write_text("original")
    gate = tmp_path / "gate.json"
    write_json(
        gate, {"gate": "G1", "status": "PASS", "input_sha256": {str(source): sha256(source)}}
    )
    require_gate(gate, "G1")
    source.write_text("changed")
    with pytest.raises(RuntimeError, match="stale"):
        require_gate(gate, "G1")
    with pytest.raises(FileExistsError):
        write_json(gate, {})


def test_budget_reserves_upload_time(monkeypatch):
    from repbench.v2 import state

    monkeypatch.setattr(
        state.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=0, stdout="Spark booking: 0:35:00 left (h:mm:ss)."
        ),
    )
    assert remaining_seconds() == 2100
    assert require_budget(100) == 2100
    with pytest.raises(RuntimeError, match="Insufficient"):
        require_budget(900)
    monkeypatch.setattr(
        state.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="no booking"),
    )
    with pytest.raises(RuntimeError, match="deadline"):
        remaining_seconds()


def test_parse_timeleft_with_days():
    from repbench.v2.state import parse_timeleft

    assert parse_timeleft("Spark booking: 1-00:35:23 left (h:mm:ss).") == 86400 + 35 * 60 + 23
    assert parse_timeleft("Spark booking: 2:35:12 left (h:mm:ss).") == 2 * 3600 + 35 * 60 + 12
