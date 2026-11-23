"""A100 v2 path on synthetic A100-like fixtures (no real A100 data, no real models).

Covers preparation (v1 label semantics, white compositing, Polyvore-format round trip,
refuse-overwrite, archive checksum), the frozen-protocol guard, probe scoring with the
pipeline's fitted-probe code path (selection cross-checked against a pipeline run),
summary statistics/CSV schema, make_tables_v2 consumption, and the MLLM runner's A100
loading and guard.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import random
import subprocess
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from repbench.data.polyvore import PolyvoreDisjoint
from repbench.v2.a100 import (
    A100_MARGINS,
    load_questions,
    mean_cluster_bootstrap,
    prepare_a100_v2,
    question_records,
    require_frozen_protocol,
    summarize_block,
)
from repbench.v2.probe_scoring import fit_valid_carveout

ROOT = Path(__file__).resolve().parents[1]
DIM = 48
LAT_NUMS = list(range(1, 13))
AAT_NUMS = [1, 2, 21, 22, 53, 54, 68, 69, 80, 81, 92, 93]  # two per facet
MODELS = ("resnet50", "siglip2_so400m_384")
NOISE = {"resnet50": 2.2, "siglip2_so400m_384": 0.8}


def _jpeg(colour):
    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buffer, format="JPEG")
    return buffer.getvalue()


def _rgba_png():
    image = Image.new("RGBA", (8, 8), (0, 0, 0, 0))
    image.putpixel((0, 0), (200, 10, 10, 255))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_archive(path: Path):
    """Archive laid out like the official A100.zip; returns item id -> latent style."""
    prefix = "AiDLab/A100"
    styles, entries = {}, {}
    rng = random.Random(7)
    lat_rows, aat_rows = [], []
    for n in LAT_NUMS:
        style = n % 4
        question = [f"Cat_Pq{n}i{j}" for j in range(1 + n % 3)]
        answers = [f"Top_Pq{n}a{k}" for k in range(5)]
        majority = n % 5
        archive = majority if n % 4 else (majority + 1) % 5  # some disagreements
        dist = [0.1] * 5
        dist[majority] = 0.4 + (0.1 if archive == majority else 0.0)
        if archive != majority:
            dist[archive] = 0.3
        total = sum(dist)
        dist = [d / total for d in dist]
        lat_rows.append({"quesion_num": n, "question": question, "answers": answers,
                         "gt": str(archive + 1), "gt_distribution": dist,
                         "question_count": len(question)})
        for ref in question:
            styles["LAT_" + ref.split("_", 1)[1]] = style
        for k, ref in enumerate(answers):
            other = (style + 1 + k % 3) % 4  # never the gold style
            styles["LAT_" + ref.split("_", 1)[1]] = style if k == majority else other
        for ref in question + answers:
            entries[f"{prefix}/LAT/image/sub/{ref.split('_', 1)[1]}.jpg"] = _jpeg(
                (rng.randrange(256), 0, 0))
    for n in AAT_NUMS:
        style = n % 4
        question = [f"Clothings/Outwear_{n:03d}Q0{j}" for j in range(3 + n % 2)]
        answers = [f"Shoes/Heels_{n:03d}A0{k}" for k in range(5)]
        gt = (n * 3) % 5
        aat_rows.append({"quesion_num": n, "question": question, "answers": answers,
                         "gt": gt + 1, "question_count": len(question)})
        for ref in question:
            styles["AAT_" + ref.rsplit("_", 1)[1]] = style
        for k, ref in enumerate(answers):
            other = (style + 1 + k % 3) % 4  # never the gold style
            styles["AAT_" + ref.rsplit("_", 1)[1]] = style if k == gt else other
        for ref in question + answers:
            entries[f"{prefix}/AAT/image/{ref.rsplit('_', 1)[1]}.jpg"] = _jpeg(
                (0, rng.randrange(256), 0))
    # One transparent PNG stored under a .jpg name (must be composited on white).
    entries[f"{prefix}/AAT/image/{AAT_NUMS[0]:03d}Q00.jpg"] = _rgba_png()
    entries[f"{prefix}/LAT/image/unreferenced.jpg"] = _jpeg((1, 2, 3))
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(f"{prefix}/LAT/label/LAT.json", json.dumps(lat_rows))
        archive.writestr(f"{prefix}/AAT/label/AAT.json", json.dumps(aat_rows))
        archive.writestr(f"__MACOSX/{prefix}/LAT/label/._LAT.json", b"junk")
        archive.writestr(f"{prefix}/.DS_Store", b"junk")
        for name, data in entries.items():
            archive.writestr(name, data)
    return styles


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@pytest.fixture()
def prepared(tmp_path):
    archive = tmp_path / "A100.zip"
    styles = make_archive(archive)
    out = tmp_path / "a100_v2"
    manifest = prepare_a100_v2(archive, out, expected_sha256=_sha(archive))
    return archive, out, manifest, styles


def test_prepare_semantics_and_files(prepared, tmp_path):
    archive, out, manifest, _ = prepared
    assert manifest["archive_sha256"] == _sha(archive)
    assert [manifest["tasks"][t]["questions"] for t in ("LAT", "AAT")] == [12, 12]
    questions, _ = load_questions(out)
    lat = [q for q in questions if q.task == "LAT"]
    aat = [q for q in questions if q.task == "AAT"]
    for q in lat:  # primary = majority (v1), archive kept as sensitivity
        assert q.primary_gt_index == q.majority_gt_index == int(np.argmax(q.gt_distribution))
    disagreements = [q.question_num for q in lat if q.archive_gt_index != q.majority_gt_index]
    assert disagreements == manifest["lat_archive_majority_disagreements"] == [4, 8, 12]
    assert {q.facet for q in aat} == {"Color", "Style", "Occasion", "Season", "Material",
                                      "Balance"}
    assert all(q.primary_gt_index == q.archive_gt_index for q in aat)
    # White compositing of the transparent image; JPEG bytes untouched.
    mapping = json.loads((out / "image_paths.json").read_text())
    png_item = f"AAT_{AAT_NUMS[0]:03d}Q00"
    assert mapping[png_item].endswith(".png") and manifest["composited_items"] == [png_item]
    with Image.open(out / "images" / mapping[png_item]) as image:
        assert image.mode == "RGB" and image.getpixel((5, 5)) == (255, 255, 255)
    with zipfile.ZipFile(archive) as z:
        raw = z.read("AiDLab/A100/LAT/image/sub/Pq1a0.jpg")
    assert (out / "images" / "LAT_Pq1a0.jpg").read_bytes() == raw
    # Polyvore-format split 'test' resolves to the same questions; no CP examples.
    data = PolyvoreDisjoint(out.parent, out, "a100")
    fitb = data.fitb("test")
    assert [f.candidate_item_ids for f in fitb] == [q.candidate_items for q in questions]
    assert [f.correct_index for f in fitb] == [q.primary_gt_index for q in questions]
    assert data.compatibility("test") == []
    for name, digest in manifest["files"].items():
        assert _sha(out / name) == digest
    # Refuse overwrite; wrong archive checksum refused before anything is written.
    with pytest.raises(FileExistsError):
        prepare_a100_v2(archive, out, expected_sha256=None)
    with pytest.raises(ValueError, match="sha256"):
        prepare_a100_v2(archive, tmp_path / "other", expected_sha256="0" * 64)
    assert not (tmp_path / "other").exists() and not (tmp_path / "other.partial").exists()


def _hash_file(tmp_path):
    protocol = tmp_path / "v2_protocol.yaml"
    protocol.write_text("frozen: true\n")
    record = tmp_path / "v2_protocol_hash.json"
    record.write_text(json.dumps({"file": str(protocol), "sha256": _sha(protocol)}))
    return record, protocol


def test_frozen_protocol_guard(tmp_path):
    with pytest.raises(PermissionError, match="frozen"):
        require_frozen_protocol(tmp_path / "missing.json")
    record, protocol = _hash_file(tmp_path)
    assert require_frozen_protocol(record)["sha256"] == _sha(protocol)
    protocol.write_text("frozen: false\n")
    with pytest.raises(PermissionError, match="changed"):
        require_frozen_protocol(record)


# ------------------------------------------------------------------ corpus + embeddings


def write_corpus(protocol: Path):
    """Synthetic Polyvore-format corpus: compatible items share a latent style."""
    protocol.mkdir(parents=True)
    for split, n in (("train", 120), ("valid", 40)):
        outfits, groups, pos, neg, fitb = [], {}, [], [], []
        for k in range(n):
            sid = f"{split[:2]}{k}"
            outfits.append({"set_id": sid, "items": [
                {"item_id": f"{sid}x{j}s{k % 4}", "index": j + 1} for j in range(3)]})
            groups[sid] = f"user{k % 7}"
        for k, outfit in enumerate(outfits):
            sid, other = outfit["set_id"], outfits[(k + 1) % n]["set_id"]
            pos.append(f"1 {sid}_1 {sid}_2 {sid}_3")
            neg.append(f"0 {sid}_1 {other}_2 {sid}_3")
            wrong = [outfits[(k + d) % n]["set_id"] for d in (1, 2, 3)]
            fitb.append({"question": [f"{sid}_1", f"{sid}_2"],
                         "answers": [f"{sid}_3"] + [f"{w}_3" for w in wrong]})
        (protocol / f"{split}.json").write_text(json.dumps(outfits))
        (protocol / f"compatibility_{split}.txt").write_text("\n".join(pos + neg) + "\n")
        (protocol / f"fill_in_blank_{split}.json").write_text(json.dumps(fitb))
        (protocol / f"group_ids_{split}.json").write_text(json.dumps(groups))
    return sorted({i["item_id"] for s in ("train", "valid")
                   for o in json.loads((protocol / f"{s}.json").read_text()) for i in o["items"]})


def write_store(folder: Path, ids, styles, model):
    centers = np.random.default_rng(1).normal(size=(4, DIM))
    seed = int.from_bytes(hashlib.sha256(f"{model}:{folder.parent.name}".encode()).digest()[:4],
                          "little")
    gen = np.random.default_rng(seed)
    values = np.stack([centers[styles[i]] + NOISE[model] * gen.normal(size=DIM) for i in ids])
    folder.mkdir(parents=True)
    np.save(folder / "items.npy", values.astype(np.float32))
    (folder / "item_ids.json").write_text(json.dumps(list(ids)))
    (folder / "metadata.json").write_text(json.dumps(
        {"items_sha256": _sha(folder / "items.npy"), "model": model, "embedding_dim": DIM}))


def run(args, **kwargs):
    return subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, capture_output=True,
                          text=True, check=False, **kwargs)


def test_score_and_summarize_end_to_end(prepared, tmp_path):
    _, a100, _, a100_styles = prepared
    protocol = tmp_path / "synthetic_clean"
    corpus_ids = write_corpus(protocol)
    emb = tmp_path / "emb"
    for model in MODELS:
        write_store(emb / "synthetic_clean" / model, corpus_ids,
                    {i: int(i[-1]) for i in corpus_ids}, model)
        write_store(emb / "a100" / model, sorted(a100_styles), a100_styles, model)
    # Pipeline run (valid) on the corpus: reference for the refit selection.
    pipeline_results = tmp_path / "pipeline"
    done = run([ROOT / "scripts/v2/pipeline.py", "--datasets",
                f"synthetic_clean={protocol}=user", "--models", ",".join(MODELS),
                "--embeddings", emb, "--results", pipeline_results, "--settings", "native",
                "--learners", "logistic", "--operators", "abs_product", "--replicates", "50"])
    assert done.returncode == 0, done.stderr
    score = [ROOT / "scripts/v2/a100_v2.py", "score", "--a100", a100, "--embeddings", emb,
             "--corpora", f"synthetic_clean={protocol}", "--models", ",".join(MODELS),
             "--output", tmp_path / "scores", "--verify-pipeline-results", pipeline_results,
             "--selection-split", "carveout"]
    # Frozen-protocol guard: refused without the hash file, nothing written.
    refused = run([*score, "--frozen-protocol-hash", tmp_path / "absent.json"])
    assert refused.returncode != 0 and "frozen" in refused.stderr
    assert not (tmp_path / "scores").exists()
    record, _ = _hash_file(tmp_path)
    done = run([*score, "--frozen-protocol-hash", record])
    assert done.returncode == 0, done.stderr
    payload = json.loads((tmp_path / "scores/predictions/"
                          "synthetic_clean__siglip2_so400m_384__native__logistic.json").read_text())
    assert payload["meta"]["pipeline_selection_verified"] is True
    records = payload["records"]
    assert len(records) == 24
    lat = [r for r in records if r["task"] == "LAT"]
    assert all(r["correct_majority"] in (0, 1) and r["human_agreement"] is not None for r in lat)
    assert all(r["correct_majority"] is None and r["facet"] for r in records if r["task"] == "AAT")
    assert np.mean([r["correct_primary"] for r in records]) > 0.5  # chance 0.2
    # Re-running reuses verified outputs (no overwrite).
    again = run([*score, "--frozen-protocol-hash", record])
    assert again.returncode == 0 and "reuse" in again.stdout

    # Synthetic zero-shot MLLM A100 runs (format of scripts/v2/mllm_zeroshot.py).
    from repbench.v2.registry import ZERO_SHOT

    questions, _ = load_questions(a100)
    mllm_dirs = []
    for task in ("LAT", "AAT"):
        folder = tmp_path / f"mllm_{task}"
        folder.mkdir()
        (folder / "config.json").write_text(json.dumps(
            {"source": "a100", "split": task, "model": ZERO_SHOT.checkpoint}))
        (folder / "summary.json").write_text("{}")
        lines = []
        for q in (q for q in questions if q.task == task):
            permutation = [4, 3, 2, 1, 0]
            logprobs = {letter: -float(i + 1) for i, letter in enumerate("ABCDE")}
            lines.append(json.dumps({
                "question_id": q.question_id, "permutation": permutation,
                "letter_logprobs": logprobs, "predicted_index": permutation[0],
                "correct_index": q.primary_gt_index}))
        (folder / "results.jsonl").write_text("\n".join(lines) + "\n")
        mllm_dirs.append(folder)

    summary = tmp_path / "summary"
    base = [ROOT / "scripts/v2/a100_v2.py", "summarize", "--a100", a100, "--scores",
            tmp_path / "scores", "--output", summary, "--replicates", "200"]
    refused = run([*base, "--frozen-protocol-hash", tmp_path / "absent.json"])
    assert refused.returncode != 0 and not summary.exists()
    done = run([*base, "--frozen-protocol-hash", record, "--mllm", *mllm_dirs])
    assert done.returncode == 0, done.stderr
    family = summary / "synthetic_clean__native__logistic"
    metrics = pd.read_csv(family / "metrics.csv")
    assert {"dataset", "model", "setting", "learner", "split", "metric", "value", "ci_low",
            "ci_high", "training_corpus", "selection"} <= set(metrics.columns)
    assert set(metrics.dataset) == {"a100_lat", "a100_aat"}
    assert {"acc_majority", "acc_archive", "mlat", "acc", "acc_color", "acc_balance"} <= set(
        metrics.metric)
    assert (metrics.ci_low <= metrics.value).all() and (metrics.value <= metrics.ci_high).all()
    pairwise = pd.read_csv(family / "pairwise.csv")
    binary = pairwise[pairwise.metric != "mlat"]
    assert binary.mcnemar_exact_p.notna().all() and "bootstrap_p_holm" in pairwise
    equivalence = pd.read_csv(family / "equivalence.csv")
    assert sorted(equivalence.margin.unique()) == list(A100_MARGINS)
    zero = pd.read_csv(summary / "zero_shot" / "metrics.csv")
    assert set(zero.model) == {ZERO_SHOT.model_key}
    assert set(zero.dataset) == {"a100_lat", "a100_aat"}
    # MLLM letter scores map back to archive order: chosen = original candidate 4.
    mllm_questions = pd.read_csv(summary / "zero_shot" / "questions.csv")
    assert (mllm_questions.chosen_index == 4).all()
    assert (summary / "metrics_all.csv").stat().st_size > 0
    # make_tables_v2 consumes a family directory.
    generated = tmp_path / "generated"
    done = run([ROOT / "scripts/v2/make_tables_v2.py", "--results", family, "--out", generated])
    assert done.returncode == 0, done.stderr
    assert "a100" in (generated / "table_main.tex").read_text()
    again = run([*base, "--frozen-protocol-hash", record])  # summary exists: refuse
    assert again.returncode != 0 and "Refusing to overwrite" in again.stderr


def test_carveout_is_deterministic_and_disjoint(tmp_path):
    protocol = tmp_path / "c"
    write_corpus(protocol)
    examples = PolyvoreDisjoint(protocol.parent, protocol, "c").compatibility("train")
    first = fit_valid_carveout(examples, "c", 20260912)
    second = fit_valid_carveout(examples, "c", 20260912)
    assert first == second
    train_cp, fit_cp, info = first
    assert info == {"train_pairs": 96, "fit_valid_pairs": 24}
    assert len(train_cp) + len(fit_cp) == len(examples)
    assert not {e.item_ids for e in train_cp} & {e.item_ids for e in fit_cp}


def test_summarize_block_single_model_and_mlat_bootstrap(prepared):
    _, a100, _, _ = prepared
    questions, _ = load_questions(a100)
    scores = [[float(k == q.primary_gt_index) for k in range(5)] for q in questions]
    chosen = [q.primary_gt_index for q in questions]
    records = question_records(questions, scores, chosen, {"model": "m"})
    metrics, pairwise, _, _ = summarize_block({"m": records}, ["m"], {"setting": "s"},
                                              replicates=100)
    by_metric = {r["metric"]: r for r in metrics}
    assert by_metric["acc_majority"]["value"] == 1.0 and by_metric["acc"]["value"] == 1.0
    assert by_metric["acc_archive"]["value"] == pytest.approx(9 / 12)
    lat = [q for q in questions if q.task == "LAT"]
    assert by_metric["mlat"]["value"] == pytest.approx(
        np.mean([max(q.gt_distribution) for q in lat]))
    assert pairwise == []
    boot = mean_cluster_bootstrap(np.array([[0.2], [0.4], [0.9]]), groups=["a", "b", "c"],
                                  replicates=50)
    assert boot["samples"].shape == (50, 1) and 0.2 <= boot["samples"].min()


def _mllm_script():
    spec = importlib.util.spec_from_file_location("mllm_script_a100",
                                                  ROOT / "scripts/v2/mllm_zeroshot.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mllm_runner_reads_prepared_a100_and_is_guarded(prepared, tmp_path):
    _, a100, _, _ = prepared
    script = _mllm_script()
    for task in ("LAT", "AAT"):
        args = Namespace(source="a100", split=task, a100_root=a100, images_root=None)
        source, questions = script.load_all_questions(args)
        assert source == a100 / f"{task.lower()}.json" and len(questions) == 12
        assert all(Path(p).is_file() for q in questions
                   for p in (*q.partial_images, *q.candidate_images))
        assert all(len(q.candidate_images) == 5 for q in questions)
        ids = tmp_path / f"ids_{task}.json"
        script.main(["freeze-ids", "--source", "a100", "--a100-root", str(a100), "--split", task,
                     "--output", str(ids)])
        frozen = json.loads(ids.read_text())
        assert frozen["question_ids"] == [q.question_id for q in questions]
    lat = {q.question_id: q for q in script.load_all_questions(
        Namespace(source="a100", split="LAT", a100_root=a100, images_root=None))[1]}
    rows = {f"LAT_{r['question_num']}": r for r in json.loads((a100 / "lat.json").read_text())}
    assert all(lat[k].correct_index == rows[k]["majority_gt_index"] for k in lat)
    png = [p for q in script.load_all_questions(Namespace(
        source="a100", split="AAT", a100_root=a100, images_root=None))[1]
        for p in q.partial_images if str(p).endswith(".png")]
    assert len(png) == 1
    score = ["score", "--ids", str(tmp_path / "ids_LAT.json"), "--a100-root", str(a100),
             "--output-dir", str(tmp_path / "run")]
    with pytest.raises(SystemExit, match="protocol freeze"):
        script.main(score)
    with pytest.raises(SystemExit, match="frozen"):
        script.main([*score, "--confirm-protocol-frozen",
                     "--frozen-protocol-hash", str(tmp_path / "absent.json")])
    assert not (tmp_path / "run").exists()
