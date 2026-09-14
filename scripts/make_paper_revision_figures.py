"""Regenerate manuscript figures from frozen publication tables only."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9.5, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
    "axes.spines.top": False, "axes.spines.right": False,
})

KEYS = ["resnet50", "dinov3_vitl16", "clip_vitl14_336", "siglip2_b16_384",
        "fashionclip2", "marqo_fashionsiglip", "gr_lite"]
NAMES = {"resnet50":"ResNet50", "dinov3_vitl16":"DINOv3", "clip_vitl14_336":"CLIP",
         "siglip2_b16_384":"SigLIP2", "fashionclip2":"FashionCLIP",
         "marqo_fashionsiglip":"Marqo", "gr_lite":"GR-Lite"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]
GRAY = np.linspace(.15, .72, 7)

def save(fig, stem):
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

main = pd.read_csv(ROOT / "reports/tables/final/main_polyvore_iqon_results.csv")

# Fig. 1: clean cross-dataset replication, with offsets chosen to keep labels distinct.
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05))
offsets = {"resnet50":(5,-10), "dinov3_vitl16":(5,5), "clip_vitl14_336":(-33,7),
           "siglip2_b16_384":(5,-11), "fashionclip2":(-49,-10),
           "marqo_fashionsiglip":(5,6), "gr_lite":(5,-9)}
for ax, metric, ylabel in zip(axes, ["cp_auc", "fitb_acc"], ["CP ROC--AUC", "FITB accuracy"]):
    p = main[main.dataset.eq("polyvore_d_clean")].set_index("representation")
    q = main[main.dataset.eq("iqon3000_clean")].set_index("representation")
    for i, k in enumerate(KEYS):
        ax.scatter(p.loc[k, metric], q.loc[k, metric], marker=MARKERS[i], s=38,
                   facecolor=str(GRAY[i]), edgecolor="black", linewidth=.5, zorder=3)
        ax.annotate(NAMES[k], (p.loc[k, metric], q.loc[k, metric]), xytext=offsets[k],
                    textcoords="offset points", fontsize=7.2)
    lo = min(p[metric].min(), q[metric].min())-.012
    hi = max(p[metric].max(), q[metric].max())+.012
    ax.plot([lo, hi], [lo, hi], ls="--", lw=.8, color="0.65")
    ax.set(xlabel=f"Polyvore-D-Clean {ylabel}", ylabel=f"IQON3000-Clean {ylabel}")
    ax.grid(color="0.9", lw=.5)
fig.tight_layout(w_pad=2.0); save(fig, "Fig1")

# Fig. 2: A100 external corroboration.
a = pd.read_csv(ROOT / "reports/tables/final/a100_headline.csv")
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
x = np.arange(7); width = .36
for ax, src, title in zip(axes, ["polyvore_d_clean", "iqon3000_clean"],
                          ["Probe trained on Polyvore-D-Clean", "Probe trained on IQON3000-Clean"]):
    s = a[a.training_source.eq(src)]
    for j, task in enumerate(["LAT", "AAT"]):
        vals = [float(s[(s.representation.eq(k)) & (s.task.eq(task))].accuracy.iloc[0]) for k in KEYS]
        ax.bar(x+(j-.5)*width, vals, width, color="white" if j == 0 else "0.55",
               edgecolor="black", hatch="///" if j == 0 else "", linewidth=.6, label=task)
    ax.axhline(.2, ls="--", lw=.8, color="0.55", label="Chance" if src=="polyvore_d_clean" else None)
    ax.set_title(title); ax.set_xticks(x, [NAMES[k] for k in KEYS], rotation=35, ha="right")
    ax.set_ylim(0, .75); ax.grid(axis="y", color="0.9", lw=.5)
axes[0].set_ylabel("Five-choice accuracy")
axes[0].legend(frameon=False, ncol=3, loc="upper left")
fig.tight_layout(); save(fig, "Fig2")

# Fig. 3: exact LookBench checkpoint overlap; rank is 1=best.
over = ["dinov3_vitl16", "clip_vitl14_336", "siglip2_b16_384", "marqo_fashionsiglip", "gr_lite"]
lb = {"dinov3_vitl16":43.97, "clip_vitl14_336":39.79, "siglip2_b16_384":59.44,
      "marqo_fashionsiglip":62.77, "gr_lite":65.71}
cols = [("LookBench\nFine R@1", lb),
        ("Polyvore\nCP", dict(zip(over, main[(main.dataset.eq("polyvore_d_clean")) & main.representation.isin(over)].set_index("representation").loc[over].cp_auc))),
        ("Polyvore\nFITB", dict(zip(over, main[(main.dataset.eq("polyvore_d_clean")) & main.representation.isin(over)].set_index("representation").loc[over].fitb_acc))),
        ("IQON\nCP", dict(zip(over, main[(main.dataset.eq("iqon3000_clean")) & main.representation.isin(over)].set_index("representation").loc[over].cp_auc))),
        ("IQON\nFITB", dict(zip(over, main[(main.dataset.eq("iqon3000_clean")) & main.representation.isin(over)].set_index("representation").loc[over].fitb_acc)))]
fig, ax = plt.subplots(figsize=(7.0, 3.25)); xx=np.arange(len(cols))
for i,k in enumerate(over):
    ranks=[]
    for _,d in cols:
        order=sorted(d, key=d.get, reverse=True); ranks.append(order.index(k)+1)
    ax.plot(xx, ranks, marker=MARKERS[KEYS.index(k)], color=str(GRAY[KEYS.index(k)]),
            markeredgecolor="black", markeredgewidth=.4, lw=1.3, label=NAMES[k])
ax.set_xticks(xx,[c[0] for c in cols]); ax.set_yticks(range(1,6)); ax.invert_yaxis()
ax.set_ylabel("Rank (1 = best)"); ax.grid(axis="y", color="0.9", lw=.5)
ax.legend(frameon=False, ncol=5, loc="lower center", bbox_to_anchor=(.5,1.01))
fig.tight_layout(); save(fig, "Fig3")

# Fig. 4: reliability diagrams, friendly names and line styles.
rel = pd.read_csv(ROOT / "artifacts/final/calibration/reliability_bins_15.csv")
fig, axes = plt.subplots(1,2,figsize=(7.2,3.25),sharex=True,sharey=True)
for ax, ds, title in zip(axes,["polyvore_d_clean","iqon3000_clean"],["Polyvore-D-Clean","IQON3000-Clean"]):
    ax.plot([0,1],[0,1],ls="--",lw=.9,color="black",label="Ideal")
    for i,k in enumerate(KEYS):
        s=rel[(rel.dataset.eq(ds)) & (rel.representation.eq(k)) & (rel["count"]>0)]
        ax.plot(s.mean_probability,s.empirical_positive_rate,marker=MARKERS[i],ms=2.8,
                lw=.85,color=str(GRAY[i]),label=NAMES[k])
    ax.set_title(title); ax.set_xlabel("Mean predicted probability"); ax.grid(color="0.92",lw=.5)
axes[0].set_ylabel("Empirical positive rate")
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5,1.01), fontsize=8)
fig.tight_layout(rect=(0,0,1,.86)); save(fig,"Fig4")

# Fig. 5: IQON Pareto view only; the table provides full cost reporting.
eff=pd.read_csv(ROOT / "reports/tables/final/model_characteristics_and_efficiency.csv").set_index("representation")
q=main[main.dataset.eq("iqon3000_clean")].set_index("representation")
fig,axes=plt.subplots(1,2,figsize=(7.2,3.25))
off={"resnet50":(4,5),"dinov3_vitl16":(4,5),"clip_vitl14_336":(-35,5),"siglip2_b16_384":(4,-10),
     "fashionclip2":(4,5),"marqo_fashionsiglip":(-35,5),"gr_lite":(4,-10)}
for ax,metric,label in zip(axes,["cp_auc","fitb_acc"],["CP ROC--AUC","FITB accuracy"]):
    for i,k in enumerate(KEYS):
        xval=eff.loc[k,"estimated_gflops_per_image"]; yval=q.loc[k,metric]
        ax.scatter(xval,yval,s=38,marker=MARKERS[i],facecolor=str(GRAY[i]),edgecolor="black",lw=.5)
        ax.annotate(NAMES[k],(xval,yval),xytext=off[k],textcoords="offset points",fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("Estimated GFLOPs/image (log scale)"); ax.set_ylabel(f"IQON {label}")
    ax.grid(color="0.9",lw=.5)
fig.tight_layout(); save(fig,"Fig5")

# Fig. 6: all construction-seed outcomes (mean +/- SD).
cs=pd.read_csv(ROOT / "reports/tables/final/construction_seed_stability.csv")
fig,axes=plt.subplots(2,2,figsize=(7.2,5.0),sharex=True)
for ax,(ds,metric,title) in zip(axes.ravel(),[("polyvore_d_clean","cp_auc","Polyvore CP"),("polyvore_d_clean","fitb_accuracy","Polyvore FITB"),("iqon3000_clean","cp_auc","IQON CP"),("iqon3000_clean","fitb_accuracy","IQON FITB")]):
    s=cs[(cs.dataset.eq(ds)) & (cs.metric.eq(metric))].set_index("representation")
    vals=[s.loc[k,"mean"] for k in KEYS]; errs=[s.loc[k,"sd"] for k in KEYS]
    ax.bar(x,vals,yerr=errs,color=[str(v) for v in GRAY],edgecolor="black",linewidth=.5,capsize=2)
    ax.set_title(title); ax.grid(axis="y",color="0.9",lw=.5)
for ax in axes[1]: ax.set_xticks(x,[NAMES[k] for k in KEYS],rotation=35,ha="right")
fig.tight_layout(); save(fig,"Fig6")

# Fig. 7: PCA-256/native width.
pn=pd.read_csv(ROOT / "reports/tables/final/iqon_pca256_vs_native.csv").set_index("representation")
fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
for ax,short,label in zip(axes,["cp","fitb"],["CP ROC--AUC","FITB accuracy"]):
    for i,k in enumerate(KEYS):
        ax.plot([0,1],[pn.loc[k,f"pca256_{short}_{'auc' if short=='cp' else 'acc'}"],pn.loc[k,f"native_{short}_{'auc' if short=='cp' else 'acc'}"]],
                marker=MARKERS[i],color=str(GRAY[i]),lw=1.1,label=NAMES[k])
    ax.set_xticks([0,1],["PCA-256","Native width"]); ax.set_ylabel(f"IQON {label}"); ax.grid(color="0.9",lw=.5)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5,1.01), fontsize=8)
fig.tight_layout(rect=(0,0,1,.86)); save(fig,"Fig7")

# Fig. 8: compatibility-label efficiency; PCA retains all train-item embeddings.
ld=pd.read_csv(ROOT / "reports/tables/final/low_data_results.csv")
ld=ld[(ld.dataset.eq("polyvore_d_clean")) & (ld.learner.eq("logistic"))]
fig,axes=plt.subplots(1,2,figsize=(7.2,3.25))
for ax,metric,label in zip(axes,["cp_auc","fitb_acc"],["CP ROC--AUC","FITB accuracy"]):
    for i,k in enumerate(KEYS):
        s=ld[ld.representation.eq(k)].sort_values("train_fraction")
        ax.plot(100*s.train_fraction,s[metric],marker=MARKERS[i],color=str(GRAY[i]),lw=1.1,label=NAMES[k])
    ax.set_xlabel("Compatibility-labelled training outfits (%)"); ax.set_ylabel(label); ax.grid(color="0.9",lw=.5)
handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(.5,1.01), fontsize=8)
fig.tight_layout(rect=(0,0,1,.86)); save(fig,"Fig8")

# Fig. 9: AAT facet accuracy, two training sources, shared uncluttered colorbar.
ad=pd.read_csv(ROOT / "reports/tables/final/a100_aat_dimensions.csv")
facets=["Color","Style","Occasion","Season","Material","Balance"]
fig,axes=plt.subplots(1,2,figsize=(7.5,3.4),sharey=True)
im=None
for ax,src,title in zip(axes,["polyvore_d_clean","iqon3000_clean"],["Polyvore-trained","IQON-trained"]):
    arr=np.array([[ad[(ad.training_source.eq(src)) & ad.representation.eq(k) & ad.task.eq(f"AAT-{f}")].accuracy.iloc[0] for f in facets] for k in KEYS])
    im=ax.imshow(arr,vmin=0,vmax=1,cmap="Greys",aspect="auto")
    ax.set_xticks(range(6),facets,rotation=35,ha="right"); ax.set_title(title)
    for i in range(7):
        for j in range(6): ax.text(j,i,f"{arr[i,j]:.2f}",ha="center",va="center",fontsize=6.2,color="white" if arr[i,j]>.58 else "black")
axes[0].set_yticks(range(7),[NAMES[k] for k in KEYS]); axes[1].tick_params(labelleft=False)
fig.colorbar(im,ax=axes.ravel().tolist(),fraction=.025,pad=.035,label="Accuracy")
fig.subplots_adjust(left=.14,right=.90,bottom=.22,top=.88,wspace=.08); save(fig,"Fig9")

print(f"Wrote revised vector figures to {OUT}")
