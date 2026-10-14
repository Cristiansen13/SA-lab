# ============================================================
# Statistical Analysis - Employee Dataset
# ============================================================

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no window pop-ups)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import dcor
import warnings
warnings.filterwarnings("ignore")

# ── Plot style ───────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f1117",
    "axes.facecolor":   "#1a1d27",
    "axes.edgecolor":   "#3a3d4a",
    "axes.labelcolor":  "#e0e0e0",
    "xtick.color":      "#a0a0b0",
    "ytick.color":      "#a0a0b0",
    "text.color":       "#e0e0e0",
    "grid.color":       "#2a2d3a",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "sans-serif",
})
ACCENT = ["#7c6ff7", "#f76f7c", "#6fd1f7", "#f7c96f"]

# ============================================================
# 1. MERGE
# ============================================================
df_ext = pd.read_csv("employees_extended.csv")
df_rec = pd.read_csv("employees_recovered.csv")
df     = pd.merge(df_ext, df_rec, on="employee_id", how="inner")

# ============================================================
# 2. ORDINAL ENCODING
# ============================================================
work_mode_cats = ["remote", "hybrid", "onsite"]
contract_cats  = ["contractor", "part_time", "full_time"]
education_cats = ["high_school", "bachelor", "master", "phd"]
seniority_cats = ["junior", "mid", "senior", "lead"]

ordinal_cols  = ["work_mode", "contract_type", "education_level", "seniority"]
ordinal_order = [work_mode_cats, contract_cats, education_cats, seniority_cats]

enc = OrdinalEncoder(categories=ordinal_order, dtype=np.int64)
df[ordinal_cols] = enc.fit_transform(df[ordinal_cols])
for col in ordinal_cols:
    df[col] = df[col].astype("int8")

# ============================================================
# 3. CORRELATION MATRICES
# ============================================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
df_num   = df[num_cols]

pearson_corr  = df_num.corr(method="pearson")
spearman_corr = df_num.corr(method="spearman")
kendall_corr  = df_num.corr(method="kendall")

# Distance correlation – pairwise (NaN-safe)
n        = len(num_cols)
dcor_mat = pd.DataFrame(np.ones((n, n)), index=num_cols, columns=num_cols)
for i, ci in enumerate(num_cols):
    for j, cj in enumerate(num_cols):
        if i == j:
            continue
        if j < i:
            dcor_mat.iloc[i, j] = dcor_mat.iloc[j, i]
            continue
        mask = df_num[[ci, cj]].dropna().index
        x    = df_num.loc[mask, ci].values.astype(float)
        y    = df_num.loc[mask, cj].values.astype(float)
        dcor_mat.iloc[i, j] = dcor.distance_correlation(x, y)

# ── Print matrices ───────────────────────────────────────────
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.float_format", "{:.3f}".format)

print("\n===== PEARSON =====")
print(pearson_corr)
print("\n===== SPEARMAN =====")
print(spearman_corr)
print("\n===== KENDALL =====")
print(kendall_corr)
print("\n===== DISTANCE CORRELATION =====")
print(dcor_mat.round(3))

# ── Save heatmaps ────────────────────────────────────────────
def plot_heatmap(ax, matrix, title, cmap="RdBu_r", vmin=-1, vmax=1):
    mask = np.zeros_like(matrix, dtype=bool)
    mask[np.triu_indices_from(mask, k=1)] = True
    sns.heatmap(matrix, ax=ax, mask=mask, cmap=cmap, vmin=vmin, vmax=vmax,
                annot=True, fmt=".2f", annot_kws={"size": 5},
                linewidths=0.3, linecolor="#0f1117",
                cbar_kws={"shrink": 0.7}, square=True)
    ax.set_title(title, color="#e0e0e0", fontsize=11, pad=8)
    ax.tick_params(axis="x", rotation=45, labelsize=6)
    ax.tick_params(axis="y", rotation=0,  labelsize=6)

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle("Correlation Matrices", color="#e0e0e0", fontsize=15, y=1.01)
plot_heatmap(axes[0, 0], pearson_corr,  "Pearson")
plot_heatmap(axes[0, 1], spearman_corr, "Spearman")
plot_heatmap(axes[1, 0], kendall_corr,  "Kendall")
plot_heatmap(axes[1, 1], dcor_mat,      "Distance Correlation",
             cmap="YlOrRd", vmin=0, vmax=1)
plt.tight_layout()
plt.savefig("correlation_matrices.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.close()

# ============================================================
# 4. SAMPLING DISTRIBUTIONS – seniority vs annual_bonus
# ============================================================
N_SAMPLES    = 1_000
SAMPLE_SIZES = [25, 60]
METHODS      = ["pearson", "spearman", "kendall", "distance"]
COLORS       = dict(zip(METHODS, ACCENT))

rng     = np.random.default_rng(42)
pair_df = df[["seniority", "annual_bonus"]].dropna().reset_index(drop=True)
results = {}

for sz in SAMPLE_SIZES:
    results[sz] = {m: [] for m in METHODS}
    for _ in range(N_SAMPLES):
        idx    = rng.choice(len(pair_df), size=sz, replace=True)
        sample = pair_df.iloc[idx]
        x = sample["seniority"].values.astype(float)
        y = sample["annual_bonus"].values.astype(float)
        sx, sy = pd.Series(x), pd.Series(y)
        results[sz]["pearson"].append(sx.corr(sy, method="pearson"))
        results[sz]["spearman"].append(sx.corr(sy, method="spearman"))
        results[sz]["kendall"].append(sx.corr(sy, method="kendall"))
        results[sz]["distance"].append(dcor.distance_correlation(x, y))

# ── Print stats table ────────────────────────────────────────
print("\n===== SAMPLING DISTRIBUTION STATISTICS =====")
print(f"{'Method':12s} {'n':>4s}  {'Mean':>8s}  {'Std':>8s}  {'2.5%':>8s}  {'97.5%':>8s}")
print("-" * 56)
for sz in SAMPLE_SIZES:
    for m in METHODS:
        vals = np.array(results[sz][m])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        print(f"{m:12s} {sz:>4d}  {vals.mean():>8.4f}  {vals.std():>8.4f}"
              f"  {lo:>8.4f}  {hi:>8.4f}")
    print()

# ── Spread interpretation ────────────────────────────────────
print("===== SPREAD REDUCTION (n=25 -> n=60) =====")
print(f"{'Method':12s}  {'std(n=25)':>10s}  {'std(n=60)':>10s}  {'reduction':>10s}")
print("-" * 48)
for m in METHODS:
    s25 = np.std(results[25][m])
    s60 = np.std(results[60][m])
    print(f"{m:12s}  {s25:>10.4f}  {s60:>10.4f}  {100*(1-s60/s25):>9.1f}%")

# ── Save sampling distribution plots ────────────────────────
fig, axes = plt.subplots(len(METHODS), len(SAMPLE_SIZES),
                         figsize=(14, 4 * len(METHODS)), sharey="row")
fig.suptitle(
    "Sampling Distribution of Correlation Coefficients\n"
    "(seniority vs annual_bonus  |  1 000 repetitions)",
    color="#e0e0e0", fontsize=13, y=1.01
)
for col_idx, sz in enumerate(SAMPLE_SIZES):
    for row_idx, m in enumerate(METHODS):
        ax    = axes[row_idx, col_idx]
        vals  = np.array(results[sz][m])
        color = COLORS[m]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        ax.hist(vals, bins=40, color=color, alpha=0.85,
                edgecolor="#0f1117", linewidth=0.4)
        ax.axvline(vals.mean(), color="white",  lw=1.8, ls="-",
                   label=f"mean={vals.mean():.3f}")
        ax.axvline(lo,          color="#ffd06f", lw=1.2, ls="--",
                   label=f"2.5%={lo:.3f}")
        ax.axvline(hi,          color="#ffd06f", lw=1.2, ls="--",
                   label=f"97.5%={hi:.3f}")
        ax.set_title(f"{m.capitalize()}  |  n={sz}   std={vals.std():.4f}",
                     fontsize=9, color="#e0e0e0")
        ax.set_xlabel("Correlation coefficient", fontsize=8)
        ax.set_ylabel("Frequency" if col_idx == 0 else "", fontsize=8)
        ax.legend(fontsize=6.5, framealpha=0.3)
        ax.tick_params(labelsize=7)
plt.tight_layout()
plt.savefig("sampling_distributions.png", dpi=150, bbox_inches="tight",
            facecolor="#0f1117")
plt.close()

print("\nDone. Plots saved: correlation_matrices.png, sampling_distributions.png")
