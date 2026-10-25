"""
analyze.py - builds every publication table / figure from the finished runs.

    python3 analyze.py --exp_root /home/krishna_19622/dataset/experiments

Outputs (exp_root/analysis/):
  Table1_summary.csv/.tex         Dice, HD95, precision, recall per configuration
  Table2_paired_tests.csv/.tex    main vs every other config: Wilcoxon signed-rank, Holm-corrected, effect sizes, bootstrap CI
  Table3_energy_spiking.csv/.tex  firing rate, MACs, SOPs, estimated energy per slice
  Fig_dice_boxplot / Fig_paired_differences / Fig_dice_vs_lesion_size
  Fig_qualitative (DWI-only vs DWI+ADC vs ANN)  /  Fig_spiking  /  Fig_training_curves
Each figure is saved as PDF (vector) and PNG (300 dpi).
"""
import os, csv, argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--exp_root", default="/home/krishna_19622/dataset/experiments")
ap.add_argument("--out", default=None)
ap.add_argument("--main", default="snn_dwi_adc_T2_k3_s42_lopo")
ap.add_argument("--baseline", default="ann_dwi_adc_T1_k3_s42_lopo")
ap.add_argument("--dwi_only", default="snn_dwi_T2_k3_s42_lopo")
ap.add_argument("--n_qual", type=int, default=4)
args = ap.parse_args()
OUT = args.out or os.path.join(args.exp_root, "analysis")
os.makedirs(OUT, exist_ok=True)
E_MAC, E_AC = 4.6e-12, 0.9e-12

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "DejaVu Sans"})
rng = np.random.default_rng(0)


def savefig(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def label(n):
    parts = n.split("_")
    model = parts[0].upper()
    mod = "DWI+ADC" if "adc" in n else "DWI"
    T = [x for x in parts if x.startswith("T") and x[1:].isdigit()][0][1:]
    k = [x for x in parts if x.startswith("k") and x[1:].isdigit()][0][1:]
    return f"{model} {mod} (k={k})" if model == "ANN" else f"SNN {mod} (T={T}, k={k})"


def readcsv(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def fl(x):
    try:
        return float(x)
    except Exception:
        return np.nan


# ---------------------------------------------------------------- load
runs = {}
for n in sorted(os.listdir(args.exp_root)):
    if "_pp" in n:  # post-processing variants are analysed separately (pp_select.py)
        continue
    d = os.path.join(args.exp_root, n)
    if os.path.exists(os.path.join(d, "ALL_DONE")):
        rows = {}
        for r in readcsv(os.path.join(d, "per_patient.csv")):
            rows[r["patient"]] = r  # last row wins (dedupe)
        runs[n] = rows
print("complete runs:", list(runs))
if args.main not in runs:
    raise SystemExit(f"main run {args.main} not finished yet")
common = sorted(set.intersection(*[set(v) for v in runs.values()]))
print("patients common to all runs:", len(common))
# patients whose ground-truth mask is EMPTY in the public data have an undefined Dice -> reported separately, excluded from statistics
empty = [p for p in common if fl(runs[args.main][p]["gt_vox"]) == 0]
common = [p for p in common if p not in empty]
print(f"excluded {len(empty)} patients with empty ground truth: {empty}; statistics use n={len(common)}")
open(os.path.join(OUT, "excluded_empty_gt.txt"), "w").write("\n".join(empty) + "\n")


def arr(run, key, pats=None):
    pats = pats or common
    return np.array([fl(runs[run][p][key]) for p in pats])


def boot_ci(x, n=10000):
    x = x[np.isfinite(x)]
    b = [rng.choice(x, len(x)).mean() for _ in range(n)]
    return np.percentile(b, [2.5, 97.5])


def tex_table(path, header, rows, caption):
    with open(path, "w") as f:
        f.write("\\begin{table}[t]\\centering\\caption{%s}\n\\begin{tabular}{%s}\\hline\n" % (caption, "l" + "c" * (len(header) - 1)))
        f.write(" & ".join(header) + " \\\\ \\hline\n")
        for r in rows:
            f.write(" & ".join(str(x).replace("_", "\\_").replace("%", "\\%") for x in r) + " \\\\\n")
        f.write("\\hline\\end{tabular}\\end{table}\n")


def write_table(name, header, rows, caption):
    with open(os.path.join(OUT, name + ".csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    tex_table(os.path.join(OUT, name + ".tex"), header, rows, caption)
    print("saved", name)


order = [args.main] + [r for r in runs if r != args.main]

# ---------------------------------------------------------------- Table 1
rows = []
for n in order:
    d, h, pr, rc = arr(n, "dice"), arr(n, "hd95"), arr(n, "precision"), arr(n, "recall")
    lo, hi = boot_ci(d)
    q1, q3 = np.percentile(d, [25, 75])
    rows.append([label(n), len(d), f"{d.mean():.3f} ± {d.std(ddof=1):.3f}", f"{np.median(d):.3f} [{q1:.3f}-{q3:.3f}]",
                 f"{lo:.3f}-{hi:.3f}", f"{np.nanmean(h):.2f} ± {np.nanstd(h, ddof=1):.2f}",
                 f"{np.nanmean(pr):.3f}", f"{np.nanmean(rc):.3f}"])
write_table("Table1_summary", ["Model", "N", "Dice mean±SD", "Dice median [IQR]", "95% CI (mean)", "HD95 (mm) mean±SD",
                              "Precision", "Recall"], rows,
            "Patient-wise cross-validated segmentation performance (HD95 over patients with non-empty prediction).")

# ---------------------------------------------------------------- Table 2: paired tests
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run_max = 0
    for i, idx in enumerate(o):
        run_max = max(run_max, (len(p) - i) * p[idx]); adj[idx] = min(1.0, run_max)
    return adj


tests = []
ANN_DWI = "ann_dwi_T1_k3_s42_lopo"
pairs = [(args.main, n) for n in order[1:]]
if ANN_DWI in runs:  # completes the 2x2 design (SNN/ANN x DWI/DWI+ADC)
    pairs += [(args.baseline, ANN_DWI), (args.dwi_only, ANN_DWI)]
pairs = [(x, y) for x, y in pairs if x in runs and y in runs and x != y]
for (m1, n) in pairs:
    for key in ["dice", "hd95"]:
        a, b = arr(m1, key), arr(n, key)
        ok = np.isfinite(a) & np.isfinite(b)
        diff = (a - b)[ok]
        if len(diff) < 5:
            continue
        if np.allclose(diff, 0):
            pval, r_rb = 1.0, 0.0
        else:
            pval = stats.wilcoxon(diff).pvalue
            nz = diff[diff != 0]; rk = stats.rankdata(np.abs(nz))
            r_rb = (rk[nz > 0].sum() - rk[nz < 0].sum()) / rk.sum()
        dz = diff.mean() / (diff.std(ddof=1) + 1e-12)
        bs = [rng.choice(diff, len(diff)).mean() for _ in range(5000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        hi_better = key == "dice"  # Dice: higher is better, HD95: lower is better
        first_better = int((diff > 0).sum()) if hi_better else int((diff < 0).sum())
        second_better = int((diff < 0).sum()) if hi_better else int((diff > 0).sum())
        tests.append([n, key, len(diff), diff.mean(), lo, hi, pval, r_rb, dz, first_better, second_better, m1])
padj = {}
for key in ["dice", "hd95"]:
    idx = [i for i, t in enumerate(tests) if t[1] == key]
    if idx:
        for i, v in zip(idx, holm([tests[i][6] for i in idx])):
            padj[i] = v
rows = []
for i, t in enumerate(tests):
    sig = "***" if padj[i] < 0.001 else "**" if padj[i] < 0.01 else "*" if padj[i] < 0.05 else "ns"
    rows.append([f"{label(t[11])} vs {label(t[0])}", t[1].upper(), t[2], f"{t[3]:+.4f}", f"[{t[4]:+.4f}, {t[5]:+.4f}]",
                 f"{t[6]:.2e}", f"{padj[i]:.2e} ({sig})", f"{t[7]:+.2f}", f"{t[8]:+.2f}", f"{t[9]}/{t[10]}"])
write_table("Table2_paired_tests", ["Comparison", "Metric", "n", "Mean diff", "95% CI of diff", "Wilcoxon p", "Holm-adj. p",
                                    "Rank-biserial r", "Cohen dz", "first better / second better"], rows,
            "Paired comparisons (difference = first - second; for HD95 a negative difference favours the first model). Holm correction within each metric.")
if len(runs) >= 3:
    fr = stats.friedmanchisquare(*[arr(n, "dice") for n in order])
    print(f"Friedman test over {len(order)} configurations (Dice): chi2={fr.statistic:.2f}, p={fr.pvalue:.2e}")
    open(os.path.join(OUT, "friedman.txt"), "w").write(f"chi2={fr.statistic:.3f} p={fr.pvalue:.3e} n_configs={len(order)} n_patients={len(common)}\n")

# ---------------------------------------------------------------- Fig: boxplot
fig, ax = plt.subplots(figsize=(6.7, 3.0))
data = [arr(n, "dice") for n in order]
bp = ax.boxplot(data, showfliers=False, patch_artist=True, widths=0.55)
for b in bp["boxes"]:
    b.set(facecolor="#dbe9f6", edgecolor="#2b5d8a")
for med in bp["medians"]:
    med.set(color="#c0392b")
for i, d in enumerate(data):
    ax.scatter(rng.normal(i + 1, 0.06, len(d)), d, s=2, color="k", alpha=0.18, linewidths=0)
    ax.plot(i + 1, d.mean(), "D", color="#1e8449", ms=4)
ax.set_xticks(range(1, len(order) + 1)); ax.set_xticklabels([label(n).replace(" (", "\n(") for n in order], rotation=0, fontsize=6)
ax.set_ylabel("Patient-wise Dice"); ax.set_ylim(-0.02, 1.02)
savefig(fig, "Fig_dice_boxplot")

# ---------------------------------------------------------------- Fig: paired differences
comp = [n for n in [args.baseline, args.dwi_only] if n in runs]
if comp:
    fig, axs = plt.subplots(1, len(comp), figsize=(3.3 * len(comp), 2.6), squeeze=False)
    for ax, n in zip(axs[0], comp):
        diff = arr(args.main, "dice") - arr(n, "dice")
        ax.hist(diff, bins=40, color="#7fb3d5", edgecolor="white")
        ax.axvline(0, color="k", lw=0.8); ax.axvline(np.median(diff), color="#c0392b", ls="--", lw=1)
        ax.set_xlabel(f"Dice difference per patient\n({label(args.main)} - {label(n)})", fontsize=6.5); ax.set_ylabel("Patients")
    savefig(fig, "Fig_paired_differences")

# ---------------------------------------------------------------- Fig: Dice vs lesion size
g, d = arr(args.main, "gt_vox"), arr(args.main, "dice")
fig, ax = plt.subplots(figsize=(3.4, 2.8))
ax.scatter(g + 1, d, s=6, alpha=0.5, color="#2b5d8a", linewidths=0)
edges = np.quantile(g, np.linspace(0, 1, 7)); cx, cy = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (g >= lo) & (g <= hi)
    if m.sum():
        cx.append(np.median(g[m]) + 1); cy.append(np.median(d[m]))
ax.plot(cx, cy, "-o", color="#c0392b", ms=3, lw=1, label="binned median")
ax.set_xscale("log"); ax.set_xlabel("Lesion size (voxels, 2 mm in-plane)"); ax.set_ylabel("Dice"); ax.legend(frameon=False)
savefig(fig, "Fig_dice_vs_lesion_size")

# ---------------------------------------------------------------- Fig: qualitative
qruns = [(args.dwi_only, "SNN DWI only"), (args.main, "SNN DWI+ADC"), (args.baseline, "ANN DWI+ADC")]
qruns = [(n, t) for n, t in qruns if n in runs and os.path.isdir(os.path.join(args.exp_root, n, "qual"))]  # finished runs only
if qruns:
    pats = sorted(common, key=lambda p: fl(runs[args.main][p]["dice"]))
    pats = [p for p in pats if all(os.path.exists(os.path.join(args.exp_root, n, "qual", p + ".npz")) for n, _ in qruns + [(args.main, "")])]
    sel = [pats[int(q * (len(pats) - 1))] for q in np.linspace(0.9, 0.1, args.n_qual)]
    cols = 3 + len(qruns)
    fig, axs = plt.subplots(len(sel), cols, figsize=(1.35 * cols, 1.4 * len(sel)), squeeze=False)
    for r, pid in enumerate(sel):
        m0 = np.load(os.path.join(args.exp_root, args.main, "qual", pid + ".npz"))
        brain = np.abs(m0["dwi"].astype(float)) > 0
        xs, ys = np.where(brain)
        x0, x1, y0, y1 = max(xs.min() - 4, 0), xs.max() + 4, max(ys.min() - 4, 0), ys.max() + 4
        sl = (slice(x0, x1), slice(y0, y1))
        show = lambda a: a[sl].T

        def panel(ax, bg, gt=None, pr=None, title=None):
            ax.imshow(show(bg), cmap="gray", origin="lower", interpolation="nearest")
            if gt is not None and gt[sl].any():
                ax.contour(show(gt.astype(float)), levels=[0.5], colors="lime", linewidths=0.8)
            if pr is not None and pr[sl].any():
                ax.contour(show(pr.astype(float)), levels=[0.5], colors="red", linewidths=0.8)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if title:
                ax.set_title(title, fontsize=5.5, pad=2)
        gt = m0["gt"]
        panel(axs[r, 0], m0["dwi"].astype(float), title="DWI" if r == 0 else None)
        panel(axs[r, 1], m0["adc"].astype(float), title="ADC" if r == 0 else None)
        panel(axs[r, 2], m0["dwi"].astype(float), gt=gt, title="Ground truth" if r == 0 else None)
        axs[r, 0].set_ylabel(pid.replace("sub-strokecase", "#"), fontsize=6)
        for c, (n, t) in enumerate(qruns):
            m = np.load(os.path.join(args.exp_root, n, "qual", pid + ".npz"))
            panel(axs[r, 3 + c], m0["dwi"].astype(float), gt=gt, pr=m["pred"])
            axs[r, 3 + c].set_title((t + "\n" if r == 0 else "") + f"Dice {float(m['dice']):.2f}", fontsize=5.5, pad=2)
    fig.text(0.5, -0.01, "green = ground truth, red = prediction (largest-lesion axial slice; Dice = whole-volume Dice of that patient)", ha="center", fontsize=5.5)
    savefig(fig, "Fig_qualitative")

# ---------------------------------------------------------------- Spiking analysis
def energy_row(n):
    ss = readcsv(os.path.join(args.exp_root, n, "spike_stats.csv"))
    ss = [r for r in ss if r["patient"] in common]
    sl = sum(int(r["n_slices"]) for r in ss)
    macs, sops = sum(fl(r["macs"]) for r in ss), sum(fl(r["sops"]) for r in ss)
    return macs / sl, sops / sl, (macs * E_MAC + sops * E_AC) / sl


def rates(n):
    lr = readcsv(os.path.join(args.exp_root, n, "layer_rates.csv"))
    lr = [r for r in lr if r["patient"] in common]
    sp, ct, order_l = {}, {}, []
    for r in lr:
        if r["layer"] not in sp:
            order_l.append(r["layer"]); sp[r["layer"]] = 0.0; ct[r["layer"]] = 0.0
        sp[r["layer"]] += fl(r["spikes"]); ct[r["layer"]] += fl(r["neuron_steps"])
    return order_l, np.array([sp[l] / ct[l] for l in order_l])


E = {n: energy_row(n) for n in runs if os.path.exists(os.path.join(args.exp_root, n, "spike_stats.csv"))}
ref = E.get(args.baseline)
rows = []
for n in order:
    if n not in E:
        continue
    macs, sops, en = E[n]
    fr = ""
    if n.startswith("snn"):
        fr = f"{100 * rates(n)[1].mean():.1f}%"
    rows.append([label(n), fr, f"{macs / 1e6:.1f}", f"{sops / 1e6:.1f}", f"{en * 1e6:.2f}",
                 f"{en / ref[2]:.2f}x" if ref else "-"])
write_table("Table3_energy_spiking", ["Model", "Mean firing rate", "Dense MACs/slice (M)", "SOPs/slice (M)",
                                      "Est. energy/slice (uJ)", "Energy vs ANN"], rows,
            "Spiking activity and estimated energy per 128x128 slice (E_MAC = 4.6 pJ, E_AC = 0.9 pJ; ignores memory access and membrane updates).")

fig, axs = plt.subplots(1, 3, figsize=(6.9, 2.5))
if args.main.startswith("snn"):
    ln, rt = rates(args.main)
    axs[0].bar(range(len(ln)), 100 * rt, color="#5499c7")
    axs[0].set_xticks(range(len(ln))); axs[0].set_xticklabels(ln, rotation=90, fontsize=5)
    axs[0].set_ylabel("Firing rate (% of neuron-steps)"); axs[0].set_title("(a) Layer-wise firing rate")
for n in order:
    if n in E:
        c = "#c0392b" if n.startswith("ann") else "#2b5d8a"
        axs[1].scatter(E[n][2] * 1e6, arr(n, "dice").mean(), color=c, s=18)
        axs[1].annotate(label(n).replace(" (", "\n("), (E[n][2] * 1e6, arr(n, "dice").mean()), fontsize=4.5, xytext=(3, 3), textcoords="offset points")
axs[1].set_xlabel("Estimated energy / slice (uJ)"); axs[1].set_ylabel("Mean Dice"); axs[1].set_title("(b) Accuracy vs energy")
tr = sorted([(int(n.split("_T")[1].split("_")[0]), n) for n in runs
             if n.startswith("snn") and "adc" in n and "_k3_" in n])
if len(tr) >= 2:
    for i, (T, n) in enumerate(tr):
        d = arr(n, "dice"); lo, hi = boot_ci(d)
        axs[2].errorbar(T, d.mean(), yerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", color="#2b5d8a", capsize=2)
    axs[2].plot([t for t, _ in tr], [arr(n, "dice").mean() for _, n in tr], "-", color="#2b5d8a", lw=0.8)
    axs[2].set_xticks([t for t, _ in tr]); axs[2].set_xlabel("Time steps T"); axs[2].set_ylabel("Mean Dice (95% CI)"); axs[2].set_title("(c) Effect of time steps")
else:
    axs[2].axis("off")
fig.tight_layout()
savefig(fig, "Fig_spiking")

# ---------------------------------------------------------------- training curves
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4))
for n, col in [(args.main, "#2b5d8a"), (args.baseline, "#c0392b"), (args.dwi_only, "#1e8449")]:
    h = readcsv(os.path.join(args.exp_root, n, "training_history.csv"))
    if not h or n not in runs:
        continue
    for ax, key in zip(axs, ["train_loss", "train_dice"]):
        folds = sorted({int(r["fold"]) for r in h})
        rows_ = [[fl(r[key]) for r in h if int(r["fold"]) == f] for f in folds]
        mx = max(len(r_) for r_ in rows_)
        M = np.array([r_ for r_ in rows_ if len(r_) == mx])  # ignore folds with an incomplete history
        x = np.arange(1, M.shape[1] + 1)
        ax.plot(x, M.mean(0), color=col, label=label(n)); ax.fill_between(x, M.mean(0) - M.std(0), M.mean(0) + M.std(0), color=col, alpha=0.2)
axs[0].set_ylabel("Training loss"); axs[1].set_ylabel("Training Dice (batch)")
for ax in axs:
    ax.set_xlabel("Epoch")
axs[1].legend(frameon=False, fontsize=5.5)
fig.tight_layout()
savefig(fig, "Fig_training_curves")
print("\nDone. Output folder:", OUT)
