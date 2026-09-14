import hashlib
import json
from pathlib import Path

import yaml

from repbench.encoders.registry import PRIMARY_ENCODERS, PRIMARY_MODEL_KEYS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_protocol_hash_and_roster_are_exact():
    document = Path("docs/final_experimental_protocol.md")
    config_path = Path("configs/final_experimental_protocol.yaml")
    recorded = json.loads(Path("artifacts/final_experimental_protocol_hash.json").read_text())
    assert digest(document) == recorded["files"][document.as_posix()]
    assert digest(config_path) == recorded["files"][config_path.as_posix()]
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    configured = tuple(row["key"] for row in config["representations"])
    assert configured == PRIMARY_MODEL_KEYS
    revisions = {row["key"]: row["revision"] for row in config["representations"]}
    assert all(revisions[key] == PRIMARY_ENCODERS[key].revision for key in configured)


def test_a100_is_locked_external_only():
    config = yaml.safe_load(Path("configs/final_experimental_protocol.yaml").read_text(encoding="utf-8"))
    policy = config["datasets"]["a100_external"]
    assert policy["train_or_tune"] is False
    assert policy["candidates"] == "unchanged"
    assert config["hard_constraints"]["a100_training_or_tuning"] == "prohibited"
