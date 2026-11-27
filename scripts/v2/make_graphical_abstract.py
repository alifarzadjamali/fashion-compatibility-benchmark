"""Render the graphical abstract from the final probe, A100 and paired results."""
import argparse
from pathlib import Path

import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import FancyBboxPatch, Rectangle
import pandas as pd
from scipy.stats import spearmanr


INK = "#193445"
TEAL = "#007D78"
OCHRE = "#AC5720"
WHITE = "#FFFFFF"
WIDTH, HEIGHT = 375, 168  # PostScript points


def read_results(root):
    metrics = pd.read_csv(root / "results_r20k/metrics.csv")
    cp = metrics.query("split == 'test' and metric == 'cp_auc'")
    blocks = cp.pivot(index=["dataset", "setting", "learner"], columns="model", values="value")
    assert blocks.shape == (39, 15) and blocks.notna().all().all()
    ranks = blocks.rank(axis=1, ascending=False)
    mean = ranks.mean()
    concordance = 12 * mean.var(ddof=0) / (len(mean)**2 - 1)

    pairs = pd.read_csv(root / "results_r20k/pairwise.csv")

    def child_minus_parent(child, parent):
        rows = pairs[pairs.left.isin([child, parent]) & pairs.right.isin([child, parent])].copy()
        rows["gain"] = rows.delta.where(rows.left == child, -rows.delta)
        return rows

    marqo = child_minus_parent("marqo_fashionsiglip", "siglip_b16_224")
    fashion = child_minus_parent("fashionclip2", "openclip_b32_laion2b")
    assert (fashion.query("dataset == 'polyvore_v2_clean'").gain < 0).all()
    gr = child_minus_parent("gr_lite", "dinov3_vitl16_336").query(
        "dataset == 'polyvore_v2_clean' and metric == 'cp_auc'"
    )
    assert gr.query("setting == 'native' and learner == 'logistic'").gain.item() < 0
    assert (gr[gr.setting.str.startswith("pca")].gain > 0).all()

    a100 = pd.read_csv(root / "a100_summary_r20k/metrics_all.csv")
    correlations = {"Experts": [], "Crowd majority": []}
    for source, group in cp.query("setting == 'native' and learner == 'logistic'").groupby("dataset"):
        for label, task, metric in [("Experts", "a100_aat", "acc"), ("Crowd majority", "a100_lat", "acc_majority")]:
            external = a100.query("training_corpus == @source and dataset == @task and metric == @metric")
            matched = group.merge(external, on="model", suffixes=("_source", "_external"), validate="one_to_one")
            assert len(matched) == 15
            correlations[label].append(spearmanr(matched.value_source, matched.value_external).statistic)
    return mean, concordance, int((marqo.gain > 0).sum()), len(marqo), correlations


def render(root, output):
    mean, w, gains, total, correlations = read_results(root)
    mpl.rcParams.update({"pdf.fonttype": 42, "font.family": "DejaVu Sans"})
    fig = Figure(figsize=(WIDTH / 72, HEIGHT / 72), facecolor=WHITE)
    ax = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    ax.set_axis_off()

    def label(x, y, words, size=8, color=INK, **style):
        ax.text(x, y, words, fontsize=size, color=color, va="center", **style)

    def box(top, bottom, words):
        ax.add_patch(FancyBboxPatch((8, top), 121, bottom-top, boxstyle="round,pad=0,rounding_size=2",
                                   edgecolor=TEAL, facecolor=WHITE, linewidth=.75))
        label(68.5, (top+bottom)/2, words, ha="center", linespacing=1.03)

    def arrow(y1, y2):
        ax.annotate("", xy=(68.5, y2), xytext=(68.5, y1),
                    arrowprops=dict(arrowstyle="-|>", color=TEAL, lw=.8,
                                    shrinkA=0, shrinkB=0, mutation_scale=5))

    for x in (9, 50, 91):
        ax.add_patch(FancyBboxPatch((x, 4), 37, 25, boxstyle="round,pad=0,rounding_size=2",
                                    facecolor="#E9ECEE", edgecolor="none"))
        label(x+18.5, 16.5, "item\nimage", 8, ha="center", linespacing=1.0)
    arrow(31, 37)
    box(38, 53, "15 frozen encoders")
    arrow(54, 61)
    box(62, 86, "Leakage-controlled splits\nPolyvore · IQON · POG")
    arrow(87, 94)
    box(95, 110, "Shared pairwise probe")
    arrow(111, 118)
    box(119, 132, "CP / FITB")

    left = 143
    ax.plot([136, 136], [6, 132], color=INK, alpha=.22, lw=.6)
    label(left, 9, f"Rankings agree across 39 settings (W = {w:.2f})", 8.4, weight="bold")
    for y, key, name in [(22, "pe_core_l14_336", "PE-Core"), (32, "siglip2_so400m_384", "SigLIP2-So400m"),
                         (42, "resnet50", "ResNet-50")]:
        rank = mean[key]
        label(left, y, name)
        ax.add_patch(Rectangle((225, y-2.5), 6.2*rank, 5, color=TEAL, lw=0))
        label(230+6.2*rank, y, f"{rank:.2f}")
    label(365, 25, "Mean rank\n(lower is better)", 7.5, ha="right", linespacing=1)

    ax.plot([left, 367], [50, 50], color=INK, alpha=.22, lw=.55)
    label(left, 58, "Fashion tuning is checkpoint-specific", 9, weight="bold")
    for y, mark, text, color in [
        (71, "↑", f"Marqo-FashionSigLIP   {gains}/{total} gains", TEAL),
        (81, "↓", "FashionCLIP 2   lower on Polyvore", OCHRE),
        (91, "↕", "GR-Lite   ordering reverses with PCA", INK),
    ]:
        label(left, y, mark, 10, color, weight="bold")
        label(left+14, y, text)

    ax.plot([left, 367], [96, 96], color=INK, alpha=.22, lw=.55)
    label(left, 104, "Compatibility AUC tracks experts more closely", 8, weight="bold")
    for y, (name, values), color in zip([116, 126], correlations.items(), [TEAL, OCHRE]):
        lo, hi = min(values), max(values)
        label(left, y, name)
        x, scale = 213, 80
        ax.plot([x, x+scale], [y, y], color=INK, alpha=.18, lw=.65)
        ax.add_patch(Rectangle((x+scale*lo, y-2.5), scale*(hi-lo), 5, lw=0, color=color))
        label(300, y, f"ρ = {lo:.2f}–{hi:.2f}")

    ax.plot([213, 293], [131, 131], color=INK, lw=.5)
    for value in (0, .5, 1):
        tick_x = 213 + 80*value
        ax.plot([tick_x, tick_x], [131, 133], color=INK, lw=.5)
        label(tick_x, 137, f"{value:g}", 7, ha="center")
    label(254, 145, "Spearman \u03c1 (range across corpora)", 7, ha="center")

    ax.add_patch(Rectangle((0, 155), WIDTH, 13, color=INK, lw=0))
    label(WIDTH/2, 161.5, "Choose encoders by validating on the outfit decision, not by retrieval rank.",
          8.4, WHITE, ha="center")
    for ext in ["pdf", "png"]:
        fig.savefig(output.with_suffix("."+ext), dpi=360, facecolor=WHITE)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Final result directory containing results_r20k")
    parser.add_argument("output", type=Path, help="Output base name for PDF and PNG")
    args = parser.parse_args()
    render(args.results, args.output)
