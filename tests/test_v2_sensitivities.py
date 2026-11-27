"""Synthetic end-to-end tests for scripts/v2/run_sensitivities.py and repbench.v2.sensitivities."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from repbench.data.clean_protocol import SPLITS, file_sha256, generate_questions
from repbench.v2 import sensitivities as S
from repbench.v2.stores import load_model_store

ROOT = Path(__file__).resolve().parents[1]
DATASET = "synthetic_clean"
MODELS = {"resnet50": 3.0, "clip_vitl14_336": 2.5, "siglip2_so400m_384": 1.8,
          "marqo_fashionclip": 2.2}
CATEGORIES = ("top", "bottom", "shoes", "bag")
SIZES = {"train": 120, "valid": 60, "test": 40}


def _write_store(folder, ids, values):
    folder.mkdir(parents=True)
    np.save(folder / "items.npy", values.astype(np.float32))
    (folder / "item_ids.json").write_text(json.dumps(ids))
    digest = hashlib.sha256((folder / "items.npy").read_bytes()).hexdigest()
    (folder / "metadata.json").write_text(json.dumps({"items_sha256": digest}))


def build_protocol(root: Path):
    """Outfits of 3-4 items with semantic categories; compatible items share a style.
    Questions from the shared generator with the builder seed rule; one identity pair."""
    protocol = root / DATASET
    protocol.mkdir(parents=True)
    identity = {}
    for index, split in enumerate(SPLITS):
        rows, groups = [], {}
        for k in range(SIZES[split]):
            set_id = f"{split[:2]}{k}"
            length = 3 + k % 2
            rows.append({"set_id": set_id, "items": [
                {"item_id": f"{set_id}x{j}s{k % 4}", "index": j + 1,
                 "semantic_category": CATEGORIES[j]} for j in range(length)]})
            groups[set_id] = f"user{k % 7}"
        for row in rows:
            for item in row["items"]:
                identity[item["item_id"]] = item["item_id"]
        if split == "train":  # two near-duplicate tops share one image identity
            identity["tr1x0s1"] = identity["tr0x0s0"] = "near:tr0x0s0"
        metadata = {i["item_id"]: {"semantic_category": i["semantic_category"]}
                    for r in rows for i in r["items"]}
        cp, fitb, _ = generate_questions(rows, metadata, 20260912 + 10_000 * (index + 1),
                                         {i: identity[i] for i in metadata})
        (protocol / f"{split}.json").write_text(json.dumps(rows, indent=2))
        (protocol / f"compatibility_{split}.txt").write_text("\n".join(cp) + "\n")
        (protocol / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb, indent=2))
        (protocol / f"group_ids_{split}.json").write_text(json.dumps(groups))
    (protocol / "manifest.json").write_text(json.dumps({"seed": 20260912}))
    (root / "identity.json").write_text(json.dumps(identity))
    items = sorted(identity)
    centers = np.random.default_rng(1).normal(size=(4, 160))
    for model, noise in MODELS.items():
        gen = np.random.default_rng(len(model))
        values = np.stack([centers[int(i[-1])] + noise * gen.normal(size=160) for i in items])
        _write_store(root / "emb" / DATASET / model, items, values)
    return protocol


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("sens")
    protocol = build_protocol(root)
    results = root / "results"
    subprocess.run([sys.executable, str(ROOT / "scripts/v2/pipeline.py"),
                    "--datasets", f"{DATASET}={protocol}=two_way",
                    "--models", ",".join(MODELS), "--embeddings", str(root / "emb"),
                    "--results", str(results), "--settings", "native,pca128",
                    "--learners", "logistic", "--operators", "abs_product",
                    "--replicates", "100"], check=True, cwd=ROOT, capture_output=True)
    config = {DATASET: {"identity": "file", "identity_file": str(root / "identity.json")}}
    (root / "dataset_config.json").write_text(json.dumps(config))
    return {"root": root, "protocol": protocol, "results": results, "config": config,
            "spec": S.DatasetSpec(DATASET, protocol, "two_way")}


@pytest.mark.skipif(
    sys.platform == "win32", reason="Linux-only: Windows file locking prevents overwriting an open .npy file"
)
def test_store_merge_and_conflicts(tmp_path):
    ids = ["a", "b", "c"]
    values = np.arange(9, dtype=np.float32).reshape(3, 3)
    _write_store(tmp_path / "d" / "m" / "trainvalid", ids[:2], values[:2])
    _write_store(tmp_path / "d" / "m" / "test", ids[1:], values[1:])
    store, provenance = load_model_store(tmp_path, "d", "m")
    assert sorted(store) == ids and np.array_equal(store["c"], values[2])
    assert set(provenance["directories"]) == {"test", "trainvalid"}
    _write_store(tmp_path / "d2" / "m" / "x", ["a"], values[:1])
    _write_store(tmp_path / "d2" / "m" / "y", ["a"], values[1:2])
    with pytest.raises(ValueError, match="conflicting"):
        load_model_store(tmp_path, "d2", "m")
    np.save(tmp_path / "d" / "m" / "test" / "items.npy", values[1:] + 1)
    with pytest.raises(ValueError, match="checksum"):
        load_model_store(tmp_path, "d", "m")


def test_selection_split_mirrors_pipeline(env):
    from repbench.data.polyvore import PolyvoreDisjoint

    data = PolyvoreDisjoint(env["protocol"].parent, env["protocol"], DATASET)
    carve, fit = S.carve_train(data, DATASET)
    assert len(carve) == SIZES["train"] - SIZES["train"] // 5 and len(fit) == 2 * (SIZES["train"] // 5)
    full, valid = S.carve_train(data, DATASET, selection_split="valid")
    assert len(full) == SIZES["train"] and len(valid) == 2 * SIZES["valid"]
    assert S.default_selection_split("test") == "valid"
    assert S.default_selection_split("valid") == "carveout"


def test_nested_fractions_and_split_guard():
    subsets = S.nested_fraction_indices(97, (0.1, 0.25, 0.5, 1.0))
    assert subsets[1.0] == list(range(97))
    assert [len(subsets[f]) for f in (0.1, 0.25, 0.5)] == [10, 25, 49]
    assert set(subsets[0.1]) < set(subsets[0.25]) < set(subsets[0.5])
    assert subsets == S.nested_fraction_indices(97, (1.0, 0.5, 0.25, 0.1))
    with pytest.raises(PermissionError):
        S.check_eval_split("test", None)


def test_label_fractions_matches_pipeline_at_full_data(env):
    out = env["root"] / "lf"
    models = ["resnet50", "siglip2_so400m_384"]
    frame = S.run_label_fractions([env["spec"]], models, env["root"] / "emb", out, split="valid",
                                  fractions=(0.25, 1.0), replicates=50)
    assert set(S.METRIC_COLUMNS) <= set(frame.columns)
    assert set(frame.fraction) == {0.25, 1.0} and frame.ci_low.notna().all()
    for model in models:
        full = json.loads((out / "predictions" / f"{DATASET}__{model}__f1__logistic.json").read_text())
        reference = json.loads((env["results"] / "predictions" /
                                f"{DATASET}__{model}__native__logistic.json").read_text())
        np.testing.assert_allclose(full["cp"]["scores"], reference["cp"]["scores"], rtol=1e-9)
        assert full["fitb"]["chosen_index"] == reference["fitb"]["chosen_index"]
    with pytest.raises(FileExistsError):  # completed outputs are never overwritten
        S.run_label_fractions([env["spec"]], models, env["root"] / "emb", out, split="valid",
                              fractions=(0.25, 1.0), replicates=50)


def test_regeneration_is_verified_deterministic_and_audited(env):
    config = env["config"][DATASET]
    a = S.regenerate_protocol(env["protocol"], env["root"] / "regen_a", 42, config=config)
    b = S.regenerate_protocol(env["protocol"], env["root"] / "regen_b", 42, config=config)
    assert all(v["reproduced"] for v in a["base_verification"].values())
    for split in SPLITS:
        assert a["splits"][split]["sha256"] == b["splits"][split]["sha256"]
        assert a["splits"][split]["collision_audit"]["passes"]
        assert (env["root"] / "regen_a" / f"{split}.json").read_bytes() == \
            (env["protocol"] / f"{split}.json").read_bytes()
    assert file_sha256(env["root"] / "regen_a" / "compatibility_test.txt") != \
        file_sha256(env["protocol"] / "compatibility_test.txt")
    with pytest.raises(FileExistsError):
        S.regenerate_protocol(env["protocol"], env["root"] / "regen_a", 42, config=config)
    with pytest.raises(ValueError, match="reproduce"):
        S.regenerate_protocol(env["protocol"], env["root"] / "regen_c", 42,
                              config={**config, "base_seed": 1})


def test_collision_audit_detects_negative_equal_to_positive(env):
    rows = json.loads((env["protocol"] / "valid.json").read_text())
    cp = (env["protocol"] / "compatibility_valid.txt").read_text().splitlines()
    fitb = json.loads((env["protocol"] / "fill_in_blank_valid.json").read_text())
    identity = json.loads((env["root"] / "identity.json").read_text())
    assert S.audit_questions(rows, cp, fitb, identity)["passes"]
    positive = next(line for line in cp if line.startswith("1"))
    index = next(i for i, line in enumerate(cp) if line.startswith("0"))
    tampered = cp[:index] + ["0" + positive[1:]] + cp[index + 1:]
    report = S.audit_questions(rows, tampered, fitb, identity)
    assert not report["passes"] and report["cp_negative_item_sets_matching_positive"] == 1


def test_construction_seeds_total_uncertainty(env):
    out = env["root"] / "cs"
    frame = S.run_construction_seeds([env["spec"]], ["resnet50", "siglip2_so400m_384"],
                                     env["root"] / "emb", out, split="valid", seeds=(42, 43, 44),
                                     configs=env["config"], replicates=50)
    assert set(frame.construction_seed) == {42, 43, 44}
    summary = pd.read_csv(out / "summary.csv")
    np.testing.assert_allclose(summary.total_var,
                               summary.between_seed_var + summary.mean_bootstrap_var)
    assert (summary.n_seeds == 3).all() and summary.seed_sd.notna().all()
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["protocols"]) == 3


def test_provenance_excludes_group(env):
    out = env["root"] / "prov"
    pairwise = S.run_provenance(env["results"], [env["spec"]], out, split="valid",
                                exclude=S.provenance_group(), replicates=50)
    models = set(pairwise.left) | set(pairwise.right)
    assert models == {"resnet50", "clip_vitl14_336", "siglip2_so400m_384"}
    assert {"native", "pca128"} == set(pairwise.setting)
    assert DATASET in json.loads((out / "friedman.json").read_text())
    assert "marqo_fashionclip" in S.provenance_group()


def test_subgroups_and_failure_consensus(env):
    out = env["root"] / "sub"
    frame = S.run_subgroups(env["results"], [env["spec"]], out, split="valid", replicates=50,
                            min_cp=10, min_fitb=10)
    assert {"outfit_length", "contains_category", "blank_category"} <= set(frame.subgroup_type)
    assert set(frame[frame.subgroup_type == "outfit_length"].subgroup) == {"3", "4"}
    consensus = pd.read_csv(out / "failure_consensus.csv")
    overall = consensus[consensus.feature == "all"]
    assert (overall.unanimous_wrong_n + overall.model_disagreement_n
            + overall.unanimous_correct_n == overall.n).all()
    by_length = consensus[(consensus.feature == "outfit_length") & (consensus.task == "FITB")]
    assert by_length.n.sum() == SIZES["valid"]
    failures = json.loads((out / "unanimous_failures.json").read_text())[DATASET]
    fitb_all = overall[overall.task == "FITB"].iloc[0]
    assert len(failures["fitb_unanimous_failures"]) == fitb_all.unanimous_wrong_n


def test_near_duplicate_exclusion(env, tmp_path):
    affected = {"test_items": {"phash_train": ["va1x0s1"], "phash_valid": []},
                "test_outfits": {"any_method": ["va1", "va2", "va5"]}}
    path = tmp_path / "affected.json"
    path.write_text(json.dumps(affected))
    out = env["root"] / "nd"
    frame = S.run_near_duplicates(env["results"], [env["spec"]], {DATASET: path}, out,
                                  split="valid", replicates=50)
    manifest = json.loads((out / "manifest.json").read_text())["datasets"][DATASET]["variants"]
    outfit = manifest["exclude_outfit:any_method"]
    assert outfit["fitb_kept"] == SIZES["valid"] - 3
    assert outfit["cp_examples_kept"] == 2 * (SIZES["valid"] - 3)
    assert manifest["exclude_item:any_method"]["cp_examples_kept"] <= outfit["cp_examples_kept"]
    assert set(frame.variant) == {"full", "exclude_outfit:any_method", "exclude_item:any_method"}
    path.write_text(json.dumps({"test_outfits": {"any_method": ["te1"]}}))
    with pytest.raises(ValueError, match="not in"):
        S.run_near_duplicates(env["results"], [env["spec"]], {DATASET: path}, tmp_path / "nd2",
                              split="valid", replicates=50)


def test_lookbench_correlation(env, tmp_path):
    shipped = json.loads((ROOT / "configs" / "lookbench_v3_scores.json").read_text())
    for value in shipped["models"].values():  # owner-transcribed (P3): exact matches only scored
        assert value["score"] is None or value.get("exact_match") is True
    placeholder = tmp_path / "placeholder.json"  # unfilled config -> no correlation invented
    placeholder.write_text(json.dumps({"metric": "unfilled", "models": {
        m: {"score": None, "exact_match": False} for m in MODELS}}))
    out = S.run_lookbench(env["results"] / "metrics.csv", placeholder, tmp_path / "lb0")
    assert out.status.str.startswith("insufficient").all()
    synthetic = {"metric": "synthetic", "models": {  # test-only numbers, monotone in quality
        m: {"score": s, "exact_match": True} for m, s in
        zip(MODELS, (10.0, 20.0, 40.0, 30.0), strict=True)}}
    (tmp_path / "scores.json").write_text(json.dumps(synthetic))
    out = S.run_lookbench(env["results"] / "metrics.csv", tmp_path / "scores.json",
                          tmp_path / "lb1", replicates=200)
    row = out[out.compatibility_metric == "cp_auc"].iloc[0]
    assert row.status == "ok" and row.n_models == 4
    assert -1 <= row.spearman_rho <= 1 and row.spearman_boot_low <= row.spearman_boot_high


def test_rank_correlation_perfect():
    result = S.rank_correlation([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12], replicates=100)
    assert result["spearman_rho"] == pytest.approx(1) and result["kendall_tau"] == pytest.approx(1)


def test_cli_all(env):
    out = env["root"] / "cli"
    affected = env["root"] / "affected_cli.json"
    affected.write_text(json.dumps({"test_outfits": {"any_method": ["va3"]}}))
    base = [sys.executable, str(ROOT / "scripts/v2/run_sensitivities.py")]
    command = base + ["all", "--results", str(env["results"]), "--embeddings",
                      str(env["root"] / "emb"), "--out", str(out),
                      "--models", "resnet50,siglip2_so400m_384", "--fractions", "0.5,1.0",
                      "--construction-seeds", "42,43", "--replicates", "30",
                      "--subgroup-replicates", "0", "--min-cp", "10", "--min-fitb", "10",
                      "--dataset-config", str(env["root"] / "dataset_config.json"),
                      "--affected", f"{DATASET}={affected}"]
    subprocess.run(command, check=True, cwd=ROOT, capture_output=True)
    for step in ("label_fractions", "construction_seeds", "provenance", "subgroups",
                 "near_duplicates", "lookbench"):
        assert (out / step / "manifest.json").exists(), step
    rerun = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=True)
    assert rerun.stdout.count("[skip]") == 6
    refused = subprocess.run(base + ["subgroups", "--results", str(env["results"]),
                                     "--out", str(env["root"] / "cli_t"), "--eval-split", "test"],
                             cwd=ROOT, capture_output=True, check=False)
    assert refused.returncode != 0 and b"frozen" in refused.stderr


def test_regenerated_protocol_keeps_near_duplicate_groups(env, tmp_path):
    """D4: seed protocols must carry the base protocol's image_equivalence record, otherwise
    pipeline.equivalence_map() silently drops the near-duplicate merge from the bootstrap."""
    import shutil

    base = tmp_path / "base"
    shutil.copytree(env["protocol"], base)
    info = {"sha256": "ab" * 32, "groups": 3}
    manifest = json.loads((base / "manifest.json").read_text())
    (base / "manifest.json").write_text(json.dumps({**manifest, "image_equivalence": info}))
    config = env["config"][DATASET]
    with_groups = S.regenerate_protocol(base, tmp_path / "seed42", 42, config=config)
    assert with_groups["image_equivalence"] == info
    saved = json.loads((tmp_path / "seed42" / "manifest.json").read_text())
    assert saved["image_equivalence"] == info
    without = S.regenerate_protocol(env["protocol"], tmp_path / "seed42_plain", 42, config=config)
    assert "image_equivalence" not in without
