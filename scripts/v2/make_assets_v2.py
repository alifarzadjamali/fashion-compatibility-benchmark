"""Additional generated paper assets for RepBench-Fashion v2 (no hand-typed numbers).

Usage:
  python scripts/v2/make_assets_v2.py --ff ~/repbench-results/dgx37/final/v2/full --out <dir>

Outputs in --out:
  table_efficiency.tex, efficiency.csv  from <ff>/profile/*.json (+ profile_qwen_flops/ for Qwen)
  table_feature_width.tex               from embedding-store metadata.json + prediction JSONs
  numbers_extra.tex, numbers_extra.json `\\res...` macros for headline values quoted in
                                        docs/PAPER_GUIDE_V2.md that numbers.tex lacks
  fig_qualitative_fitb.pdf/.png         FITB cases (2 generic-correct + 1 unanimous failure per
                                        dataset), images read from the private archives
  qualitative_fitb_manifest.json        selected questions, item ids, archive members, image SHA-256
  assets_manifest.json                  input SHA-256 per asset

Macro names follow make_tables_v2.macro_name ("res" + CamelCase, digits spelled out) so
check_paper_numbers.py --extra numbers_extra.json recognises them.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import statistics
import sys
import tarfile
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables_v2 import fmt, macro_name

SEED = 20260912
# Grid intensity for kgCO2e: UK Government GHG Conversion Factors for Company Reporting 2024,
# electricity generation, "UK electricity" (kgCO2e per kWh generated).
GRID_INTENSITY_KG_PER_KWH = 0.207
GRID_INTENSITY_SOURCE = ("UK Government GHG Conversion Factors for Company Reporting 2024, "
                         "electricity generation, 'UK electricity'")
GRID_INTENSITY_YEAR = 2024
EAGER_FLOPS_MARK = r"$^\dagger$"
HOLM_ALPHA = 0.05
DATASETS = ["polyvore_v2_clean", "iqon3000_v2_clean", "pog_v2_clean"]
DATASET_LABEL = {"polyvore_v2_clean": "Polyvore", "iqon3000_v2_clean": "IQON3000",
                 "pog_v2_clean": "POG"}
FAMILY_ORDER = ["supervised_cnn", "self_supervised_vision", "generic_vision_language",
                "fashion_vision_language", "fashion_retrieval", "mllm_embedding"]
HOME = Path.home()
POLYVORE_ZIP = HOME / "repbench-mirror/raw/polyvore_outfits.zip"
POLYVORE_MEMBER = "polyvore-outfit-dataset/polyvore_outfits/images/{}.jpg"
IQON_ZIP = HOME / "repbench-mirror/iqon_audit/IQON3000.zip"
IQON_CATALOG = (HOME / "repbench-results/dgx37/final/v2/iqon_preparation/catalog/"
                "image_audit_resolved.parquet")
POG_TAR = HOME / "repbench-mirror/raw/pog/pog_images.tar"
MIRROR_SUMS = HOME / "repbench-mirror/SHA256SUMS"
PROTOCOLS = HOME / "repbench-results/protocols"
EMB = HOME / "repbench-results/emb"
CROSS_MACHINE = HOME / "repbench-results/dgx37/cross_machine_polyvore.json"
# Layout filter for the qualitative figure (readability only; recorded in the manifest).
MAX_QUESTION_ITEMS = 4


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tex(s):
    return str(s).replace("_", r"\_")


def families():
    from repbench.v2.registry import V2_ENCODERS
    return {k: v.family for k, v in V2_ENCODERS.items()}


def model_order(models):
    fam = families()
    return sorted(models, key=lambda m: (FAMILY_ORDER.index(fam[m]) if fam.get(m) in FAMILY_ORDER
                                         else 99, m))


def tabular(caption, label, header, rows, notes=()):
    lines = ["\\begin{table}[t]\\centering\\small",
             f"\\caption{{{caption}}}\\label{{{label}}}",
             "\\begin{tabular}{l" + "r" * (len(header) - 1) + "}\\toprule",
             " & ".join(header) + " \\\\\\midrule"]
    lines += [" & ".join(r) + " \\\\" for r in rows]
    lines += ["\\bottomrule\\end{tabular}"]
    if notes:
        lines += ["\\par\\smallskip{\\footnotesize " + " ".join(notes) + "}"]
    lines += ["\\end{table}"]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- 1. efficiency
def efficiency(ff, out):
    inputs, rows = [], []
    for path in sorted((ff / "profile").glob("*.json")):
        d = json.loads(path.read_text())
        inputs.append(path)
        key = d["model_key"]
        gflops, eager = d["flops"].get("gflops_per_image"), False
        qwen = ff / "profile_qwen_flops" / f"{key}.json"
        if gflops is None and qwen.is_file():
            q = json.loads(qwen.read_text())
            if q.get("model_key") != key:
                raise ValueError(f"{qwen} is for {q.get('model_key')}")
            gflops, eager = q["gflops_per_image"], True
            inputs.append(qwen)
        e = d["energy"]
        wh = e["wh_per_1k_images"]
        rows.append({
            "model": key,
            "params_m": d["image_tower_parameters_m"],
            "gflops_per_image": gflops,
            "gflops_eager_attention": eager,
            "latency_ms_per_image_b1": d["latency"]["1"]["latency_ms_per_image_median"],
            "latency_ms_per_image_b64": d["latency"]["64"]["latency_ms_per_image_median"],
            "throughput_images_per_s_b64": d["throughput_images_per_second"],
            "allocator_peak_gb": d["allocator_peak_bytes"] / 1e9,
            "cpu_preprocess_ms_per_image": 1e3 * d["preprocessing"]["per_image_seconds"]["median"],
            "energy_wh_per_1k_images": wh,
            "energy_net_of_idle_wh_per_1k_images": e["net_of_idle_wh_per_1k_images"],
            "kgco2e_per_1k_images": wh / 1e3 * GRID_INTENSITY_KG_PER_KWH,
            "grid_intensity_kg_per_kwh": GRID_INTENSITY_KG_PER_KWH,
            "grid_intensity_source": GRID_INTENSITY_SOURCE,
            "grid_intensity_year": GRID_INTENSITY_YEAR,
            "device": d["device"], "dtype": d["dtype"], "attention": d["attention_method"],
        })
    frame = pd.DataFrame(rows).set_index("model").loc[model_order([r["model"] for r in rows])]
    frame.reset_index().to_csv(out / "efficiency.csv", index=False)
    body = []
    for m, r in frame.iterrows():
        g = "--" if pd.isna(r.gflops_per_image) else f"{r.gflops_per_image:.1f}"
        body.append([tex(m), f"{r.params_m:.1f}", g + (EAGER_FLOPS_MARK if r.gflops_eager_attention else ""),
                     f"{r.latency_ms_per_image_b1:.2f}", f"{r.latency_ms_per_image_b64:.2f}",
                     f"{r.throughput_images_per_s_b64:.0f}", f"{r.allocator_peak_gb:.2f}",
                     f"{r.cpu_preprocess_ms_per_image:.2f}", f"{r.energy_wh_per_1k_images:.3f}",
                     f"{1e6 * r.kgco2e_per_1k_images:.2f}"])
    device = sorted(set(frame.device))
    header = ["model", "Params (M)", "GFLOPs/img", "Lat.\\ b1 (ms)", "Lat.\\ b64 (ms/img)",
              "Thr.\\ (img/s)", "Peak (GB)", "CPU prep (ms/img)", "Energy (Wh/1k)",
              "CO$_2$e ($10^{-6}$\\,kg/1k)"]
    notes = [
        (f"{EAGER_FLOPS_MARK} FLOPs counted with eager attention (the SDPA counter does not support "
        "this model); timing uses SDPA."),
        ("Latency: GPU-only median per image at batch 1 and batch 64; throughput at batch 64; "
        "peak = allocator peak at batch 64 ($10^9$ bytes); CPU prep = median per-image "
        "preprocessing; energy = GPU/SoC NVML counter (gross) per 1k images."),
        (f"CO$_2$e = energy $\\times$ {GRID_INTENSITY_KG_PER_KWH} kgCO$_2$e/kWh "
        f"({GRID_INTENSITY_SOURCE.replace(chr(39) + 'UK', '``UK').replace('city' + chr(39), "city''")})."),
    ]
    (out / "table_efficiency.tex").write_text(tabular(
        "Image-encoder efficiency on " + ", ".join(tex(x) for x in device) + " (bf16). "
        "Generated by scripts/v2/make\\_assets\\_v2.py.", "tab:v2-efficiency", header, body, notes))
    return inputs, frame


# ---------------------------------------------------------------- 2. feature width
def feature_width(ff, out):
    inputs, dims = [], {}
    for path in sorted(EMB.glob("*/*/*/metadata.json")):
        meta = json.loads(path.read_text())
        dims.setdefault(path.parts[-3], set()).add(int(meta["embedding_dim"]))
        inputs.append(path)
    rows = []
    for model in model_order(dims):
        if len(dims[model]) != 1:
            raise ValueError(f"{model}: inconsistent embedding_dim {dims[model]}")
        dim = dims[model].pop()
        checked = {}
        for operator, setting, factor in (("abs_product", "native", 2),
                                          ("abs_product_sum", "native+abs_product_sum", 3)):
            for ds in DATASETS:
                pred = ff / "results/predictions" / f"{ds}__{model}__{setting}__logistic.json"
                if pred.is_file():
                    p = json.loads(pred.read_text())
                    inputs.append(pred)
                    if p["item_dim_in"] != dim or p["feature_dim"] != factor * dim:
                        raise ValueError(f"{pred}: feature_dim {p['feature_dim']} != {factor}x{dim}")
                    checked[operator] = checked.get(operator, 0) + 1
        rows.append({"model": model, "embedding_dim": dim, "probe_width": 2 * dim,
                     "probe_width_sum_operator": 3 * dim, "verified_files": checked})
    body = [[tex(r["model"]), str(r["embedding_dim"]), str(r["probe_width"]),
             str(r["probe_width_sum_operator"])] for r in rows]
    header = ["model", "Embedding dim $d$", "Probe width $[|z_i-z_j|; z_i\\odot z_j]$",
              "Width with $z_i+z_j$ (sensitivity)"]
    (out / "table_feature_width.tex").write_text(tabular(
        "Embedding dimension and pair-feature width of the native probe (2$d$) and of the "
        "operator sensitivity (3$d$). Generated by scripts/v2/make\\_assets\\_v2.py.",
        "tab:v2-feature-width", header, body))
    return inputs, rows


# ---------------------------------------------------------------- 3. macros
def macros(ff, out):
    numbers, inputs, sources = {}, [], {}

    def put(value, source, *parts):
        name = macro_name(*parts)
        if name in numbers:
            raise ValueError(f"Duplicate macro {name}")
        numbers[name], sources[name] = value, source

    selection_path = ff / "results_r20k/selection.json"
    selection = json.loads(selection_path.read_text())
    inputs.append(selection_path)

    # OutfitTransformer: 5-seed mean test CP-AUC and FITB per dataset x role.
    for ds in DATASETS:
        roles = {"generic": selection[ds]["generic"], "fashion": selection[ds]["fashion"],
                 "resnet": "resnet50"}
        for role, enc in roles.items():
            path = ff / "outfit_transformer" / ds / enc / "eval_test/summary.json"
            seeds = json.loads(path.read_text())["seeds"]
            inputs.append(path)
            src = str(path)
            put(fmt("cp_auc", statistics.mean(s["test_cp_auc"] for s in seeds)), src,
                "ot", ds, role, "test", "cp_auc", "mean")
            put(fmt("fitb_acc", statistics.mean(s["test_fitb_accuracy"] for s in seeds)), src,
                "ot", ds, role, "test", "fitb_acc", "mean")
            put(str(len(seeds)), src, "ot", ds, role, "seeds")

    # Holm-significant counts at 20,000 replicates.
    pw_path = ff / "results_r20k/pairwise.csv"
    pw = pd.read_csv(pw_path)
    inputs.append(pw_path)
    sig = pw.bootstrap_p_holm < HOLM_ALPHA
    put(f"{int(sig.sum()):,}", str(pw_path), "holm", "sig", "all")
    put(f"{len(pw):,}", str(pw_path), "holm", "comparisons", "all")
    put(f"{int(pw.replicates.iloc[0]):,}", str(pw_path), "holm", "replicates")
    prim = (pw.setting == "native") & (pw.learner == "logistic")
    for metric in ("cp_auc", "fitb_acc"):
        block = prim & (pw.metric == metric)
        # "primary block" in the guide = native features, logistic probe, CP AUC.
        tag = "primary" if metric == "cp_auc" else "primary fitb"
        put(f"{int((sig & block).sum()):,}", str(pw_path), "holm", "sig", tag)
        put(f"{int(block.sum()):,}", str(pw_path), "holm", "comparisons", tag)
        for ds in DATASETS:
            b = block & (pw.dataset == ds)
            put(f"{int((sig & b).sum()):,}", str(pw_path), "holm", "sig", "primary", ds, metric)
            put(f"{int(b.sum()):,}", str(pw_path), "holm", "comparisons", "primary", ds, metric)

    # LookBench rank correlation.
    lb_path = ff / "results/sensitivities/lookbench/correlations.csv"
    lb = pd.read_csv(lb_path)
    inputs.append(lb_path)
    lb = lb[lb.status == "ok"]
    put(f"{lb.spearman_rho.min():.2f}", str(lb_path), "lookbench", "spearman rho", "min")
    put(f"{lb.spearman_rho.max():.2f}", str(lb_path), "lookbench", "spearman rho", "max")
    if lb.n_models.nunique() != 1:
        raise ValueError("LookBench n differs across rows")
    put(str(int(lb.n_models.iloc[0])), str(lb_path), "lookbench", "n")

    # Cross-machine (laptop x86 vs DGX ARM) Polyvore check.
    cm = json.loads(CROSS_MACHINE.read_text())
    inputs.append(CROSS_MACHINE)
    diffs = [r["auc_diff"] for r in cm["rows"]]
    for tag, value in (("median", statistics.median(diffs)), ("max", max(diffs))):
        mant, exp = f"{value:.1e}".split("e")
        put(f"{value:.1e}".replace("e-0", "e-"), str(CROSS_MACHINE), "cross machine", tag, "abs dauc")
        put(mant, str(CROSS_MACHINE), "cross machine", tag, "abs dauc", "mantissa")
        put(str(int(exp)), str(CROSS_MACHINE), "cross machine", tag, "abs dauc", "exponent")
    put(f"{max(diffs):.4f}", str(CROSS_MACHINE), "cross machine", "max", "abs dauc", "fixed")
    put(str(len(diffs)), str(CROSS_MACHINE), "cross machine", "files")
    put(str(sum(d > 0.001 for d in diffs)), str(CROSS_MACHINE), "cross machine", "files over",
        "milli")

    # MLLM zero-shot (Qwen3-VL-8B-Instruct) per question set.
    for summary in sorted((ff / "mllm").glob("*/summary.json")):
        s = json.loads(summary.read_text())
        inputs.append(summary)
        name = summary.parent.name
        lo, hi = s["accuracy_ci95_question_bootstrap"]
        put(fmt("fitb_acc", s["accuracy"]), str(summary), "mllm", name, "acc")
        put(fmt("fitb_acc", lo), str(summary), "mllm", name, "acc", "ci low")
        put(fmt("fitb_acc", hi), str(summary), "mllm", name, "acc", "ci high")
        put(f"{s['questions']:,}", str(summary), "mllm", name, "questions")

    # A100 headline accuracies (20k-replicate summary): LAT majority label (primary), LAT
    # archive label, AAT; zero-shot MLLM and the selected probes trained on each corpus.
    a_path = ff / "a100_summary_r20k/metrics_all.csv"
    a = pd.read_csv(a_path)
    inputs.append(a_path)
    heads = [("a100_lat", "acc_majority"), ("a100_lat", "acc_archive"), ("a100_aat", "acc")]
    for split, metric in heads:
        zs = a[(a.dataset == split) & (a.training_corpus == "zero_shot") & (a.metric == metric)]
        if len(zs) == 1:
            put(fmt("acc", zs.value.iloc[0]), str(a_path), split, "zero shot", metric)
        for ds in DATASETS:
            g = a[(a.dataset == split) & (a.training_corpus == ds) & (a.metric == metric)]
            roles = {"generic": selection[ds]["generic"], "fashion": selection[ds]["fashion"],
                     "resnet": "resnet50"}
            for role, enc in roles.items():
                r = g[g.model == enc]
                if len(r) == 1:
                    put(fmt("acc", r.value.iloc[0]), str(a_path), split, ds, role, metric)
            if len(g):
                put(fmt("acc", g.value.max()), str(a_path), split, ds, "best probe", metric)

    # Guard against collisions with the main numbers.tex.
    main_json = ff / "generated_final/numbers.json"
    if main_json.is_file():
        clash = set(numbers) & set(json.loads(main_json.read_text()))
        if clash:
            raise ValueError(f"numbers_extra macros clash with numbers.json: {sorted(clash)}")
    (out / "numbers_extra.tex").write_text(
        "% Generated by scripts/v2/make_assets_v2.py; do not edit.\n"
        + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(numbers.items())))
    (out / "numbers_extra.json").write_text(json.dumps(numbers, indent=1, sort_keys=True))
    (out / "numbers_extra_sources.json").write_text(json.dumps(sources, indent=1, sort_keys=True))
    return inputs, numbers


# ---------------------------------------------------------------- 4. qualitative FITB
class Images:
    """Read item images from the private archives in memory (nothing extracted to disk)."""

    def __init__(self):
        self._pv = self._iq = None
        self._iq_map = None
        self._pog_paths = None
        self.members = {}

    def resolve(self, ds, item_ids):
        from PIL import Image

        from repbench.v2.images import to_rgb_on_white
        raw = {}
        if ds == "polyvore_v2_clean":
            self._pv = self._pv or zipfile.ZipFile(POLYVORE_ZIP)
            for i in item_ids:
                m = POLYVORE_MEMBER.format(i)
                raw[i] = (m, self._pv.read(m))
        elif ds == "iqon3000_v2_clean":
            if self._iq_map is None:
                cat = pd.read_parquet(IQON_CATALOG, columns=["item_id", "zip_entry"])
                self._iq_map = dict(zip(cat.item_id.astype(str), cat.zip_entry))
                self._iq = zipfile.ZipFile(IQON_ZIP)
            for i in item_ids:
                m = self._iq_map[str(i)]
                raw[i] = (m, self._iq.read(m))
        elif ds == "pog_v2_clean":
            paths = json.loads((PROTOCOLS / ds / "build_a/image_paths.json").read_text())
            wanted = {"mirror/" + paths[i]: i for i in item_ids}
            with tarfile.open(POG_TAR, "r:") as tar:
                while wanted and (info := tar.next()) is not None:
                    if info.name in wanted:
                        raw[wanted.pop(info.name)] = (info.name, tar.extractfile(info).read())
                    tar.members.clear()
            if wanted:
                raise KeyError(f"POG members not found: {sorted(wanted)[:5]}")
        images = {}
        for i, (member, data) in raw.items():
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                images[i] = to_rgb_on_white(im)
            self.members[(ds, i)] = {"member": member, "sha256": hashlib.sha256(data).hexdigest()}
        return images


def pick_questions(ff, ds, generic, rng):
    from repbench.data.polyvore import PolyvoreDisjoint
    protocol = PROTOCOLS / ds / "build_a"
    questions = PolyvoreDisjoint(protocol.parent, protocol, ds).fitb("test")
    raw = json.loads((protocol / "fill_in_blank_test.json").read_text())
    pred_path = ff / "results/predictions" / f"{ds}__{generic}__native__logistic.json"
    fitb = json.loads(pred_path.read_text())["fitb"]
    if fitb["correct_index"] != [q.correct_index for q in questions]:
        raise ValueError(f"{ds}: prediction/protocol FITB order mismatch")
    uf_path = ff / "results/sensitivities/subgroups/unanimous_failures.json"
    uf = json.loads(uf_path.read_text())[ds]
    gen_pos = uf["models"].index(generic)
    ok = lambda i: len(questions[i].question_item_ids) <= MAX_QUESTION_ITEMS
    correct = [i for i, c in enumerate(fitb["correct"]) if c and ok(i)]
    fails = [f for f in uf["fitb_unanimous_failures"] if ok(f["question_index"])]
    chosen = sorted(rng.sample(correct, 2))
    fail = rng.choice(fails)
    fq = questions[fail["question_index"]]
    if raw[fail["question_index"]]["question"][0].split("_", 1)[0] != fail["set_id"] or \
            fail["correct_index"] != fq.correct_index:
        raise ValueError(f"{ds}: unanimous-failure record does not match the protocol")
    cases = []
    for i in chosen:
        cases.append({"kind": "generic_correct", "question_index": i,
                      "set_id": raw[i]["question"][0].split("_", 1)[0],
                      "chosen_index": fitb["chosen_index"][i]})
    cases.append({"kind": "unanimous_failure", "question_index": fail["question_index"],
                  "set_id": fail["set_id"], "chosen_index": fail["chosen_index"][gen_pos],
                  "models_wrong": len(fail["chosen_index"]),
                  "same_wrong_choice": fail["same_wrong_choice"]})
    for c in cases:
        q = questions[c["question_index"]]
        c.update(question_item_ids=list(q.question_item_ids),
                 candidate_item_ids=list(q.candidate_item_ids), correct_index=q.correct_index,
                 blank_position=raw[c["question_index"]]["blank_position"])
    return cases, [pred_path, uf_path, protocol / "fill_in_blank_test.json",
                   protocol / "test.json"]


def qualitative(ff, out, selection):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"font.size": 7, "pdf.fonttype": 42})
    rng = random.Random(SEED)
    loader, rows, inputs, problems, record = Images(), [], [], [], {}
    for ds in DATASETS:
        generic = selection[ds]["generic"]
        try:
            cases, used = pick_questions(ff, ds, generic, rng)
            ids = sorted({i for c in cases for i in c["question_item_ids"] + c["candidate_item_ids"]})
            images = loader.resolve(ds, ids)
        except Exception as exc:  # noqa: BLE001 - report and continue with the other datasets
            problems.append(f"{ds}: {type(exc).__name__}: {exc}")
            continue
        inputs += used
        for c in cases:
            c["images"] = {i: loader.members[(ds, i)] for i in
                           c["question_item_ids"] + c["candidate_item_ids"]}
            rows.append((ds, generic, c, images))
        record[ds] = {"generic_model": generic, "cases": cases}
    if not rows:
        return inputs, record, problems
    ncols = MAX_QUESTION_ITEMS + 1 + 4
    fig, axes = plt.subplots(len(rows), ncols, figsize=(0.8 * ncols, 0.95 * len(rows)),
                             squeeze=False)
    ok_colour, chosen_colour = "#009E73", "#D55E00"  # Okabe-Ito
    for r, (ds, generic, c, images) in enumerate(rows):
        for ax in axes[r]:
            ax.axis("off")
        for k, i in enumerate(c["question_item_ids"]):
            axes[r][k].imshow(images[i])
        for k, i in enumerate(c["candidate_item_ids"]):
            ax = axes[r][MAX_QUESTION_ITEMS + 1 + k]
            ax.imshow(images[i])
            colour = (ok_colour if k == c["correct_index"]
                      else chosen_colour if k == c["chosen_index"] else None)
            if colour:
                ax.add_patch(Rectangle((0, 0), 1, 1, transform=ax.transAxes, fill=False,
                                       lw=2.2, ec=colour, clip_on=False))
            tags = (["correct"] if k == c["correct_index"] else []) + (
                ["chosen"] if k == c["chosen_index"] else [])
            ax.set_title(" / ".join(tags), fontsize=5.5, color=colour or "black", pad=1)
        kind = ("generic probe correct" if c["kind"] == "generic_correct"
                else f"all {c['models_wrong']} probes wrong")
        axes[r][0].text(-0.15, 0.5, f"{DATASET_LABEL[ds]}\n{kind}", transform=axes[r][0].transAxes,
                        ha="right", va="center", fontsize=6.5)
        axes[r][MAX_QUESTION_ITEMS].text(0.5, 0.5, "?", ha="center", va="center", fontsize=14,
                                         transform=axes[r][MAX_QUESTION_ITEMS].transAxes)
    axes[0][1].set_title("partial outfit (question)", fontsize=7, pad=8)
    axes[0][MAX_QUESTION_ITEMS + 2].set_title("candidates", fontsize=7, pad=8, loc="right")
    fig.subplots_adjust(wspace=0.08, hspace=0.35, left=0.16)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"fig_qualitative_fitb.{ext}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    (out / "qualitative_fitb_manifest.json").write_text(json.dumps({
        "seed": SEED, "selection": str(ff / "results_r20k/selection.json"),
        "rule": "per dataset: 2 test FITB questions answered correctly by the selected generic "
                "encoder (native, logistic) + 1 unanimous failure, each drawn with "
                f"random.Random({SEED}) in dataset order {DATASETS}; questions restricted to "
                f"<= {MAX_QUESTION_ITEMS} given items for layout. Chosen = selected generic "
                "probe's choice.",
        "archives": {"polyvore": str(POLYVORE_ZIP), "iqon": str(IQON_ZIP),
                     "iqon_catalog": str(IQON_CATALOG), "pog": str(POG_TAR)},
        "datasets": record, "problems": problems}, indent=1))
    return inputs, record, problems


# ---------------------------------------------------------------- main
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--ff", type=Path, required=True, help="canonical v2 full-run root")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--skip-figure", action="store_true")
    args = parser.parse_args(argv)
    ff, out = args.ff.expanduser(), args.out.expanduser()
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"script": "scripts/v2/make_assets_v2.py", "seed": SEED,
                "grid_intensity": {"kgco2e_per_kwh": GRID_INTENSITY_KG_PER_KWH,
                                   "source": GRID_INTENSITY_SOURCE, "year": GRID_INTENSITY_YEAR},
                "assets": {}}

    def record(names, inputs):
        hashes = {str(p): sha(p) for p in dict.fromkeys(inputs)}
        for name in names:
            manifest["assets"][name] = hashes
        print(f"wrote {', '.join(names)}")

    inputs, _ = efficiency(ff, out)
    record(["table_efficiency.tex", "efficiency.csv"], inputs)
    inputs, _ = feature_width(ff, out)
    record(["table_feature_width.tex"], inputs)
    inputs, numbers = macros(ff, out)
    record(["numbers_extra.tex", "numbers_extra.json"], inputs)
    if not args.skip_figure:
        selection = json.loads((ff / "results_r20k/selection.json").read_text())
        inputs, _, problems = qualitative(ff, out, selection)
        inputs = [ff / "results_r20k/selection.json", IQON_CATALOG,
                  PROTOCOLS / "pog_v2_clean/build_a/image_paths.json", MIRROR_SUMS, *inputs]
        record(["fig_qualitative_fitb.pdf", "fig_qualitative_fitb.png",
                "qualitative_fitb_manifest.json"], inputs)
        manifest["archives_sha256_from_mirror_SHA256SUMS"] = {
            line.split()[1]: line.split()[0]
            for line in MIRROR_SUMS.read_text().splitlines() if line.strip()}
        manifest["problems"] = problems
        for p in problems:
            print("PROBLEM:", p)
    (out / "assets_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"{len(numbers)} extra macros; assets -> {out}")


if __name__ == "__main__":
    main()
