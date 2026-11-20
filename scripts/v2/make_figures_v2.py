"""Paper figures for RepBench-Fashion v2, generated only from result files (no hand-typed numbers).

Each figure is produced when its inputs exist and skipped (reported) otherwise, so the
script can run on partial results. Inputs:

  <results>/metrics.csv, rank_intervals.csv, pairwise.csv, calibration.csv  (pipeline.py stats)
  <results>/sensitivities/label_fractions/metrics.csv                      (run_sensitivities.py)
  <results>/sensitivities/construction_seeds/metrics.csv
  --mllm <dir> ... : mllm_zeroshot.py output dirs with summary.json        (optional)

Model families come from repbench.v2.registry; matched parent pairs from the frozen
protocol (configs/v2_protocol.yaml). Colours are colour-blind safe (Okabe-Ito).
Outputs (PDF + PNG) and figures_manifest.json (input SHA-256 per figure) in --out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

FAMILY_COLOUR = {  # Okabe-Ito
    "supervised_cnn": "#000000",
    "self_supervised_vision": "#E69F00",
    "generic_vision_language": "#0072B2",
    "fashion_vision_language": "#D55E00",
    "fashion_retrieval": "#CC79A7",
    "mllm_embedding": "#009E73",
}
FAMILY_LABEL = {
    "supervised_cnn": "supervised CNN", "self_supervised_vision": "self-supervised vision",
    "generic_vision_language": "generic vision-language",
    "fashion_vision_language": "fashion vision-language", "fashion_retrieval": "fashion retrieval",
    "mllm_embedding": "MLLM embedding",
}
DATASET_LABEL = {"polyvore_v2_clean": "Polyvore", "iqon3000_v2_clean": "IQON3000",
                 "pog_v2_clean": "POG"}
METRIC_LABEL = {"cp_auc": "CP AUC", "fitb_acc": "FITB accuracy"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def families():
    from repbench.v2.registry import V2_ENCODERS
    return {key: spec.family for key, spec in V2_ENCODERS.items()}


def primary(metrics, setting="native", learner="logistic"):
    return metrics[(metrics.setting == setting) & (metrics.learner == learner)]


class Figures:
    def __init__(self, out):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 8, "axes.spines.top": False,
                             "axes.spines.right": False, "pdf.fonttype": 42})
        self.plt = plt
        self.out = out
        self.manifest = {}

    def save(self, fig, name, inputs):
        for ext in ("pdf", "png"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        self.plt.close(fig)
        self.manifest[name] = {str(p): sha(p) for p in inputs}
        print(f"wrote {name}")


def fig_ranking(F, metrics, fam, inputs):
    """Primary-setting CP AUC and FITB per dataset, models sorted, coloured by family, 95% CI."""
    data = primary(metrics)
    datasets = [d for d in DATASET_LABEL if d in set(data.dataset)]
    metric_names = [m for m in METRIC_LABEL if m in set(data.metric)]
    if not datasets or not metric_names:
        return print("skip fig_ranking: no primary metrics")
    fig, axes = F.plt.subplots(len(metric_names), len(datasets), squeeze=False,
                               figsize=(2.6 * len(datasets), 0.22 * data.model.nunique() * len(metric_names) + 1))
    for r, metric in enumerate(metric_names):
        for c, dataset in enumerate(datasets):
            ax = axes[r][c]
            g = data[(data.dataset == dataset) & (data.metric == metric)].sort_values("value")
            y = range(len(g))
            for yi, row in zip(y, g.itertuples()):
                colour = FAMILY_COLOUR.get(fam.get(row.model), "grey")
                err = ([[row.value - row.ci_low], [row.ci_high - row.value]]
                       if pd.notna(getattr(row, "ci_low", None)) else None)
                ax.errorbar(row.value, yi, xerr=err, fmt="o", ms=3, color=colour, elinewidth=1)
            ax.set_yticks(list(y), list(g.model), fontsize=6)
            ax.set_title(f"{DATASET_LABEL[dataset]} — {METRIC_LABEL[metric]}", fontsize=8)
    handles = [F.plt.Line2D([], [], marker="o", ls="", color=c, label=FAMILY_LABEL[f])
               for f, c in FAMILY_COLOUR.items() if f in {fam.get(m) for m in data.model}]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.02 - 0.02 * len(handles) / 3))
    fig.tight_layout()
    F.save(fig, "fig_ranking", inputs)


def fig_rank_intervals(F, ranks, fam, inputs):
    """Bootstrap rank intervals (median, 95%) for the primary setting."""
    data = ranks[(ranks.setting == "native") & (ranks.learner == "logistic")]
    panels = [(d, m) for d in DATASET_LABEL for m in METRIC_LABEL
              if len(data[(data.dataset == d) & (data.metric == m)])]
    if not panels:
        return print("skip fig_rank_intervals: no rank intervals")
    fig, axes = F.plt.subplots(1, len(panels), squeeze=False,
                               figsize=(2.4 * len(panels), 0.22 * data.model.nunique() + 1))
    for ax, (dataset, metric) in zip(axes[0], panels):
        g = data[(data.dataset == dataset) & (data.metric == metric)].sort_values(
            "rank_median", ascending=False)
        for yi, row in enumerate(g.itertuples()):
            ax.plot([row.rank_low, row.rank_high], [yi, yi],
                    color=FAMILY_COLOUR.get(fam.get(row.model), "grey"), lw=2)
            ax.plot(row.rank_median, yi, "o", ms=3, color="black")
        ax.set_yticks(range(len(g)), list(g.model), fontsize=6)
        ax.set_xlabel("rank (1 = best)")
        ax.invert_xaxis()
        ax.set_title(f"{DATASET_LABEL[dataset]} — {METRIC_LABEL[metric]}", fontsize=8)
    fig.tight_layout()
    F.save(fig, "fig_rank_intervals", inputs)


def fig_matched_pairs(F, pairwise, pairs, inputs):
    """Fashion-adapted minus generic parent (primary setting), paired-bootstrap 95% CI."""
    data = pairwise[(pairwise.setting == "native") & (pairwise.learner == "logistic")]
    rows = []
    for fashion, parent in pairs:
        for row in data.itertuples():
            if {row.left, row.right} == {fashion, parent}:
                sign = 1 if row.left == fashion else -1
                lo, hi = sorted((sign * row.ci_low, sign * row.ci_high))
                rows.append({"pair": f"{fashion} − {parent}", "dataset": row.dataset,
                                 "metric": row.metric, "delta": sign * row.delta, "lo": lo, "hi": hi,
                                 "p": row.bootstrap_p_holm})
    if not rows:
        return print("skip fig_matched_pairs: matched pairs not in pairwise.csv")
    frame = pd.DataFrame(rows)
    metric_names = [m for m in METRIC_LABEL if m in set(frame.metric)]
    fig, axes = F.plt.subplots(1, len(metric_names), squeeze=False, sharey=True,
                               figsize=(2.6 * len(metric_names) + 1.6, 0.35 * len(frame) / len(metric_names) + 1.2))
    palette = {"polyvore_v2_clean": "#0072B2", "iqon3000_v2_clean": "#E69F00", "pog_v2_clean": "#009E73"}
    for ax, metric in zip(axes[0], metric_names):
        g = frame[frame.metric == metric].reset_index(drop=True)
        labels = list(dict.fromkeys(g.pair))
        for row in g.itertuples():
            offset = list(DATASET_LABEL).index(row.dataset) * 0.2
            ax.errorbar(row.delta, labels.index(row.pair) + offset,
                        xerr=[[row.delta - row.lo], [row.hi - row.delta]], fmt="o", ms=3,
                        color=palette.get(row.dataset, "grey"), label=DATASET_LABEL.get(row.dataset))
        ax.axvline(0, color="grey", lw=0.8, ls="--")
        ax.set_yticks([i + 0.2 for i in range(len(labels))], labels, fontsize=6)
        ax.set_xlabel(f"Δ {METRIC_LABEL[metric]} (fashion − parent)")
    handles, names = axes[0][0].get_legend_handles_labels()
    unique = dict(zip(names, handles))
    fig.legend(unique.values(), unique.keys(), frameon=False, fontsize=6,
               loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    F.save(fig, "fig_matched_pairs", inputs)


def fig_settings(F, metrics, inputs):
    """CP AUC change versus the primary cell across feature settings and learners (heatmap)."""
    data = metrics[metrics.metric == "cp_auc"].copy()
    datasets = [d for d in DATASET_LABEL if d in set(data.dataset)]
    if not datasets or data.setting.nunique() * data.learner.nunique() < 2:
        return print("skip fig_settings: single setting")
    fig, axes = F.plt.subplots(1, len(datasets), squeeze=False,
                               figsize=(3.6 * len(datasets), 0.22 * data.model.nunique() + 1.5))
    for ax, dataset in zip(axes[0], datasets):
        g = data[data.dataset == dataset].copy()
        g["cell"] = g.setting + "/" + g.learner
        pivot = g.pivot_table(index="model", columns="cell", values="value")
        base = pivot.get("native/logistic")
        if base is None:
            continue
        delta = pivot.sub(base, axis=0).drop(columns="native/logistic")
        limit = max(abs(delta.min().min()), abs(delta.max().max()), 1e-3)
        image = ax.imshow(delta.values, cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
        ax.set_xticks(range(delta.shape[1]), delta.columns, rotation=60, ha="right", fontsize=6)
        ax.set_yticks(range(delta.shape[0]), delta.index, fontsize=6)
        ax.set_title(f"{DATASET_LABEL[dataset]}: Δ CP AUC vs native/logistic", fontsize=8)
        fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    F.save(fig, "fig_settings", inputs)


def fig_calibration(F, calibration, inputs):
    """ECE (15 bins) before/after temperature and Platt scaling fitted on VALID."""
    data = calibration
    datasets = [d for d in DATASET_LABEL if d in set(data.dataset)]
    if not datasets or "ece_15" not in data:
        return print("skip fig_calibration: no calibration rows")
    fig, axes = F.plt.subplots(1, len(datasets), squeeze=False,
                               figsize=(2.8 * len(datasets), 0.22 * data.model.nunique() + 1.2))
    styles = {("temperature", "before"): ("o", "#999999", "uncalibrated"),
              ("temperature", "after"): ("s", "#0072B2", "temperature"),
              ("platt", "after"): ("^", "#D55E00", "Platt")}
    for ax, dataset in zip(axes[0], datasets):
        g = data[data.dataset == dataset]
        models = sorted(g.model.unique())
        for (method, phase), (marker, colour, label) in styles.items():
            h = g[(g.method == method) & (g.phase == phase)].set_index("model")
            ax.plot([h.ece_15.get(m) for m in models], range(len(models)), marker, ms=3,
                    color=colour, ls="", label=label)
        ax.set_yticks(range(len(models)), models, fontsize=6)
        ax.set_xlabel("ECE (15 bins)")
        ax.set_title(DATASET_LABEL[dataset], fontsize=8)
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    F.save(fig, "fig_calibration", inputs)


def fig_label_fractions(F, frame, fam, inputs):
    """Primary CP AUC versus fraction of training labels (95% CI band)."""
    data = frame[frame.metric == "cp_auc"]
    datasets = [d for d in DATASET_LABEL if d in set(data.dataset)]
    if not datasets:
        return print("skip fig_label_fractions: no rows")
    fig, axes = F.plt.subplots(1, len(datasets), squeeze=False, figsize=(3 * len(datasets), 2.6))
    for ax, dataset in zip(axes[0], datasets):
        for model, g in data[data.dataset == dataset].groupby("model"):
            g = g.sort_values("fraction")
            colour = FAMILY_COLOUR.get(fam.get(model), "grey")
            ax.plot(g.fraction, g.value, "-o", ms=2, lw=1, color=colour, label=model)
            if g.ci_low.notna().all():
                ax.fill_between(g.fraction, g.ci_low, g.ci_high, color=colour, alpha=0.08)
        ax.set_xscale("log")
        ax.set_xticks(sorted(data.fraction.unique()), [f"{f:g}" for f in sorted(data.fraction.unique())])
        ax.set_xlabel("fraction of training outfits")
        ax.set_ylabel("CP AUC")
        ax.set_title(DATASET_LABEL[dataset], fontsize=8)
    axes[0][-1].legend(frameon=False, fontsize=5, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    F.save(fig, "fig_label_fractions", inputs)


def fig_construction_seeds(F, frame, inputs):
    """Spread of primary CP AUC over negative-construction seeds (per model and dataset)."""
    data = frame[frame.metric == "cp_auc"] if "metric" in frame else frame
    seed_col = "construction_seed" if "construction_seed" in data else None
    datasets = [d for d in DATASET_LABEL if d in set(data.dataset)]
    if not datasets or seed_col is None:
        return print("skip fig_construction_seeds: no rows")
    fig, axes = F.plt.subplots(1, len(datasets), squeeze=False,
                               figsize=(2.8 * len(datasets), 0.22 * data.model.nunique() + 1.2))
    for ax, dataset in zip(axes[0], datasets):
        g = data[data.dataset == dataset]
        models = sorted(g.model.unique())
        for yi, model in enumerate(models):
            values = g[g.model == model].value
            ax.plot(values, [yi] * len(values), "o", ms=2, color="#0072B2", alpha=0.7)
        ax.set_yticks(range(len(models)), models, fontsize=6)
        ax.set_xlabel("CP AUC across construction seeds")
        ax.set_title(DATASET_LABEL[dataset], fontsize=8)
    fig.tight_layout()
    F.save(fig, "fig_construction_seeds", inputs)


def fig_mllm(F, metrics, mllm_dirs, inputs):
    """Zero-shot MLLM FITB accuracy beside the probes' FITB accuracy (same test sets)."""
    rows = []
    for directory in mllm_dirs:
        summary = Path(directory) / "summary.json"
        if summary.exists():
            s = json.loads(summary.read_text())
            rows.append({"name": Path(directory).name, "accuracy": s.get("accuracy"),
                             "n": s.get("questions"), "path": summary})
    data = primary(metrics)
    data = data[data.metric == "fitb_acc"]
    fitb = [r for r in rows if r["name"] in DATASET_LABEL]
    if not fitb or data.empty:
        return print("skip fig_mllm: no MLLM summaries or FITB rows")
    fig, axes = F.plt.subplots(1, len(fitb), squeeze=False, figsize=(2.6 * len(fitb), 3.2))
    for ax, row in zip(axes[0], fitb):
        g = data[data.dataset == row["name"]].sort_values("value")
        ax.barh(range(len(g)), g.value, color="#56B4E9")
        ax.axvline(row["accuracy"], color="#D55E00", lw=1.5,
                   label=f"Qwen3-VL-8B zero-shot ({row['n']} q)")
        ax.set_yticks(range(len(g)), list(g.model), fontsize=6)
        ax.set_xlabel("FITB accuracy")
        ax.set_title(DATASET_LABEL[row["name"]], fontsize=8)
        ax.legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    F.save(fig, "fig_mllm_vs_probes", inputs + [r["path"] for r in fitb])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sensitivities", type=Path, help="default <results>/sensitivities")
    parser.add_argument("--mllm", nargs="*", default=[], help="mllm_zeroshot output dirs")
    parser.add_argument("--protocol", type=Path, default=Path("configs/v2_protocol.yaml"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    sens = args.sensitivities or args.results / "sensitivities"
    fam = families()
    pairs = yaml.safe_load(args.protocol.read_text())["roster"]["matched_parent_pairs"]
    F = Figures(args.out)

    def load(path):
        return pd.read_csv(path) if Path(path).exists() else None

    metrics_path = args.results / "metrics.csv"
    metrics = load(metrics_path)
    if metrics is None:
        raise SystemExit(f"{metrics_path} missing")
    fig_ranking(F, metrics, fam, [metrics_path])
    fig_settings(F, metrics, [metrics_path])
    for name, fn, extra in (("rank_intervals.csv", fig_rank_intervals, (fam,)),
                            ("pairwise.csv", fig_matched_pairs, (pairs,)),
                            ("calibration.csv", fig_calibration, ())):
        frame = load(args.results / name)
        if frame is None:
            print(f"skip {fn.__name__}: {name} missing")
            continue
        if fn is fig_calibration:
            fn(F, frame, [args.results / name])
        else:
            fn(F, frame, *extra, [args.results / name])
    for sub, fn, extra in (("label_fractions", fig_label_fractions, (fam,)),
                           ("construction_seeds", fig_construction_seeds, ())):
        path = sens / sub / "metrics.csv"
        frame = load(path)
        if frame is None:
            print(f"skip {fn.__name__}: {path} missing")
            continue
        fn(F, frame, *extra, [path])
    if args.mllm:
        fig_mllm(F, metrics, args.mllm, [metrics_path])
    (args.out / "figures_manifest.json").write_text(json.dumps(F.manifest, indent=1))
    print(f"{len(F.manifest)} figures -> {args.out}")


if __name__ == "__main__":
    main()
