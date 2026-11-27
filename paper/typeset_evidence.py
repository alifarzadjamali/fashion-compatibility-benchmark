"""Typeset selected fields from completed v2 reports. No model or inference code."""
import csv
import json
from pathlib import Path

PAPER = Path(__file__).resolve().parent
V2 = PAPER.parent / "research/v2"
RESULTS = V2 / "results"
PRIMARY = RESULTS / "results_r20k"
SENS = RESULTS / "results/sensitivities"
DATASETS = {"polyvore_v2_clean": "Polyvore", "iqon3000_v2_clean": "IQON", "pog_v2_clean": "POG"}
MODELS = {
    "resnet50": "ResNet-50", "dinov3_vitl16": "DINOv3 (224)", "dinov3_vitl16_336": "DINOv3 (336)",
    "gr_lite": "GR-Lite", "clip_vitl14_336": "CLIP-L", "openclip_b32_laion2b": "OpenCLIP-B/32",
    "siglip_b16_224": "SigLIP-B", "siglip2_b16_384": "SigLIP2-B", "siglip2_so400m_384": "SigLIP2-So400m",
    "pe_core_l14_336": "PE-Core", "fashionclip2": "FashionCLIP 2", "marqo_fashionclip": "Marqo-FashionCLIP",
    "marqo_fashionsiglip": "Marqo-FashionSigLIP", "qwen3_vl_emb_2b": "Qwen-Embedding-2B",
    "qwen3_vl_emb_8b": "Qwen-Embedding-8B", "qwen3_vl_8b_zeroshot": "Qwen-Instruct",
}
selection = json.loads((PRIMARY / "selection.json").read_text())
tables = []


def rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def select(records, **fields):
    return [r for r in records if all(r[k] == v for k, v in fields.items())]


def one(records, **fields):
    record, = select(records, **fields)
    return record


def decimal(value, places=3, scale=1):
    return f"{float(value) * scale:.{places}f}"


def estimate(row, scale=1, places=3):
    return "{} [{}, {}]".format(*(decimal(row[k], places, scale) for k in ("value", "ci_low", "ci_high")))


def table(key, caption, headers, data, sources):
    fmt = "l" + "r" * (len(headers)-1)
    assert data and all(len(r) == len(headers) for r in data)
    content = [f"% Source: {sources}", r"\newcommand{\show"+key+r"}{\begin{table}[htbp]\centering\small",
               r"\caption{"+caption+r"}\label{tab:"+key+"}", r"\begin{adjustbox}{max width=\linewidth}",
               r"\begin{tabular}{@{}"+fmt+r"@{}}\toprule"]
    content += [" & ".join(headers)+r"\\\midrule"]
    content += [" & ".join(map(str,r))+r"\\" for r in data]
    tables.append("\n".join(content+[r"\bottomrule\end{tabular}\end{adjustbox}\end{table}}"]))


metrics = select(rows(PRIMARY / "metrics.csv"), setting="native", learner="logistic", split="test")
pairwise = select(rows(PRIMARY / "pairwise.csv"), setting="native", learner="logistic")
audit, pairs, contextual, external, paired, robustness, calibration, retrieval = [], [], [], [], [], [], [], []
a100 = rows(RESULTS / "a100_summary_r20k/metrics_all.csv")
seeds = rows(RESULTS / "sens_eqfix/construction_seeds/summary.csv")
dups = rows(SENS / "near_duplicates/metrics.csv")
paired_mllm = rows(RESULTS / "mllm_paired/paired_mllm_vs_probes.csv")
cal = rows(PRIMARY / "calibration.csv")
for dataset, name in DATASETS.items():
    generic = selection[dataset]["generic"]
    report = json.loads((V2 / "freeze" / f"{dataset}_build_report.json").read_text())
    assert report["builds_hash_identical"] and report["audits"]["passes"]
    counts = [*report["splits"].values()]
    assert counts == [report["splits"][k] for k in ("train","valid","test")]
    audit.append([name, *[f"{v:,}" for v in counts],
                  f"{report['image_equivalence']['groups']['groups']:,}",
                  report["audits"]["image_equivalence_not_split"]["groups_spanning_splits"]])
    native = select(metrics, dataset=dataset)
    block = []
    for cp in sorted(select(native, metric="cp_auc"), key=lambda x: -float(x["value"])):
        fitb = one(native, metric="fitb_acc", model=cp["model"])
        block.append([MODELS[cp["model"]],estimate(cp),estimate(fitb,100,1)])
    assert len(block)==len(MODELS)-1
    table(name, f"{name} test results with the native logistic probe. Brackets give marginal 95\\% cluster-bootstrap confidence intervals. Rows follow test CP-AUC for inspection.", ["Encoder","CP-AUC",r"FITB (\%)"],block,"results_r20k/metrics.csv")
    for child, parent in [("fashionclip2","openclip_b32_laion2b"),("marqo_fashionsiglip","siglip_b16_224"),("gr_lite","dinov3_vitl16_336")]:
        values = []
        for metric, factor, places in [("cp_auc",1,4),("fitb_acc",100,2)]:
            row=one(pairwise,dataset=dataset,left=child,right=parent,metric=metric)
            vals=[decimal(row[k],places,factor) for k in ("delta","ci_low","ci_high")]
            values += [f"{vals[0]} [{vals[1]}, {vals[2]}]",decimal(row["bootstrap_p_holm"],3)]
        pairs.append([name,MODELS[child],*values])
    for role, model in [("CNN","resnet50"),("Fashion",selection[dataset]["fashion"]),("Generic",selection[dataset]["generic"])]:
        rolekey={"CNN":"Resnet","Fashion":"Fashion","Generic":"Generic"}[role]
        datasetkey={"Polyvore":"Polyvore","IQON":"IqonThreeZeroZeroZero","POG":"Pog"}[name]+"VTwoClean"
        cp = one(native,model=model,metric="cp_auc")
        fitb = one(native,model=model,metric="fitb_acc")
        prefix=r"\resOt"+datasetkey+rolekey+"Test"
        contextual.append([name,MODELS[model],decimal(cp["value"]),prefix+"CpAucMean",decimal(fitb["value"],1,100),prefix+"FitbAccMean"])
        vals=[one(a100,training_corpus=dataset,model=model,metric=m) for m in ["acc_majority","acc_archive","mlat","acc"]]
        external.append([name,role,*[decimal(v["value"],1,100) for v in vals]])
        q=one(paired_mllm,dataset=dataset,probe=model)
        paired.append([name,role,decimal(q["probe_acc"],1,100),decimal(q["mllm_acc"],1,100),
                       "{} [{}, {}]".format(*(decimal(q[k],1,100) for k in ["delta_probe_minus_mllm","ci_low","ci_high"])),decimal(q["holm_p"])])
    for metric in ["cp_auc","fitb_acc"]:
        s=one(seeds,dataset=dataset,model=generic,metric=metric)
        kept=one(dups,dataset=dataset,model=generic,metric=metric,variant="exclude_outfit:any_method")
        full=one(dups,dataset=dataset,model=generic,metric=metric,variant="full")
        robustness.append([name, "CP-AUC" if metric=="cp_auc" else "FITB", *[decimal(s[k],4) for k in ("seed_min","seed_max","seed_sd","total_sd")], decimal(full["value"]), decimal(kept["value"])])
    for method,phase,shown in [("temperature","before","Raw"),("temperature","after","Temperature"),("platt","after","Platt")]:
        r=one(cal,dataset=dataset,model=generic,method=method,phase=phase)
        calibration.append([name,shown,*[decimal(r[k],4) for k in ["brier","log_loss","ece_15"]]])

external.append(["External","Qwen-Instruct",*[decimal(one(a100,training_corpus="zero_shot",metric=m)["value"],1,100) for m in ["acc_majority","acc_archive","mlat","acc"]]])
overlap=select(rows(SENS/"lookbench/overlap.csv"),used="True")
for r in sorted(select(overlap,dataset="polyvore_v2_clean"),key=lambda x:-float(x["lookbench_score"])):
    retrieval.append([MODELS[r["model"]],r["lookbench_score"],*[decimal(one(overlap,model=r["model"],dataset=d)["cp_auc"]) for d in DATASETS]])
table("Audits","Retained positive outfits by partition and image-equivalence audit. The final columns count confirmed near-duplicate groups in the scanned collection and groups shared across retained partitions.",["Corpus","Training","Validation","Test","Groups","Crossing"],audit,"freeze/*_build_report.json")
table("Contrasts","Fashion checkpoint minus its matched parent. The 95\\% intervals are paired and pointwise. Each adjusted probability refers to the complete native-probe family within its corpus and metric. FITB differences use percentage points.",["Corpus","Child","CP-AUC difference",r"$p_{\rm Holm}$","FITB difference",r"$p_{\rm Holm}$"],pairs,"results_r20k/pairwise.csv")
table("Context","Task-trained outfit scorer and native logistic probe on identical frozen image features. Transformer results are means across training seeds. FITB columns give percentages.",["Corpus","Encoder","Probe AUC","OT AUC","Probe FITB","OT FITB"],contextual,"results_r20k/{metrics.csv,selection.json}; generated_assets/numbers_extra.tex")
table("External","External A100 results in percent. LAT uses either majority choice or the archive label. mLAT gives the mean share of votes for the chosen answer. AAT uses expert judgments. Source-trained rows use validation selections.",["Source","Input","Majority","Archive","mLAT","AAT"],external,"a100_summary_r20k/metrics_all.csv")
table("Paired","Matched-question FITB comparisons with Qwen-Instruct. Accuracy is a percentage and differences are probe minus zero-shot accuracy in percentage points. Brackets give paired cluster intervals.",["Corpus","Input","Probe","Zero-shot","Difference",r"$p_{\rm Holm}$"],paired,"mllm_paired/paired_mllm_vs_probes.csv")
table("Robust","Question regeneration and image-exclusion analyses for the validation-selected generic encoder. Both metrics are on the unit scale. Min. and max. range over question seeds. Total SD includes sampling variation. The last columns compare the primary test set with the outfit-level exclusion set.",["Corpus","Task","Min.","Max.","Seed SD","Total SD","Full","Filtered"],robustness,"sens_eqfix/construction_seeds/summary.csv; near_duplicates sensitivity metrics.csv")
table("Calibrated","Test calibration for SigLIP2-So400m. The calibration map is fitted on validation outfits separately for each corpus. Smaller values are better for all three measures.",["Corpus","Score","Brier","Log loss","ECE (15 bins)"],calibration,"results_r20k/calibration.csv")
table("Retrieval","Exact checkpoint overlap with LookBench v3. Rows follow its overall fine-grained retrieval score. The final columns give native-probe compatibility AUC on each clean test partition.",["Checkpoint","Retrieval","Polyvore","IQON","POG"],retrieval,"results/sensitivities/lookbench/overlap.csv")
profile = rows(RESULTS / "generated_assets/efficiency.csv")
cost = []
for r in profile:
    cost.append([MODELS[r["model"]],decimal(r["latency_ms_per_image_b1"],2),
                 decimal(r["throughput_images_per_s_b64"],0),decimal(r["allocator_peak_gb"],2),
                 decimal(r["energy_wh_per_1k_images"],3)])
table("Cost","Image extraction on GB10. Time is single-image GPU latency. Throughput, allocator peak and gross energy describe batched inference. CPU preprocessing is measured separately.",["Encoder","Time (ms)","Images/s","Peak (GB)","Wh/1k"],cost,"generated_assets/efficiency.csv")
inflation = json.loads((RESULTS/"leakage_inflation/summary.json").read_text())["aggregates"]
split_comparison = []
for r in inflation:
    scale = 100 if r["metric"] == "fitb_acc" else 1
    places = 2 if scale == 100 else 4
    split_comparison.append(["IQON" if r["dataset"] == "iqon3000_clean" else "Polyvore",
                             "FITB (pp)" if scale == 100 else "CP-AUC",r["n_encoders"],r["n_positive"],
                             *[decimal(r[k], places, scale) for k in ("mean_inflation","min_inflation","max_inflation")]])
table("Partitions","Rebuilt pilot partition minus v2 partition under the same native probe procedure. Columns give the number of evaluated encoders, the number with a positive difference, and the distribution of differences across encoders. The test examples differ between partitions.",["Corpus","Metric","Models","Positive","Mean","Minimum","Maximum"],split_comparison,"leakage_inflation/summary.json")
(PAPER/"evidence_views.tex").write_text("\n\n".join(tables)+"\n")
