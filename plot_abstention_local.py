"""
Plot abstention curves for Residual MDN vs Gaussian URM on RewardBench.
Reads from results/abstention_results.csv and saves plot locally.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── load data ──────────────────────────────────────────────────────────────────
csv_path = os.path.join(os.path.dirname(__file__), "results", "abstention_results.csv")
retain, gauss, res_var, res_mad, lab_var = [], [], [], [], []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        retain.append(float(row["Retain Percentage"].rstrip("%")))
        gauss.append(float(row["Gaussian URM"]) * 100)
        res_var.append(float(row["Residual MDN (Variance)"]) * 100)
        res_mad.append(float(row["Residual MDN (MAD)"]) * 100)
        lab_var.append(float(row["Label MDN (Variance)"]) * 100)

retain = np.array(retain)
gauss = np.array(gauss)
res_var = np.array(res_var)
res_mad = np.array(res_mad)
lab_var = np.array(lab_var)

# ── style ──────────────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig, ax = plt.subplots(figsize=(9, 5.5))
fig.patch.set_facecolor("#0f1117")
ax.set_facecolor("#1a1d27")

COLORS = {
    "gauss":   "#64b5f6",   # blue
    "res_var": "#a5d6a7",   # green (variance)
    "res_mad": "#4caf50",   # green (MAD)
    "lab":     "#ef9a9a",   # red
}

# shaded region between the two Residual MDN curves
ax.fill_between(retain, res_var, res_mad, color=COLORS["res_var"], alpha=0.15, label="_nolegend_")

# main curves
ax.plot(retain, gauss,   color=COLORS["gauss"],   lw=2.2, marker="o", ms=5, label="Gaussian URM")
ax.plot(retain, res_var, color=COLORS["res_var"],  lw=2.2, marker="s", ms=5, linestyle="--", label="Residual MDN (Variance)")
ax.plot(retain, res_mad, color=COLORS["res_mad"],  lw=2.5, marker="D", ms=5, label="Residual MDN (MAD)")
ax.plot(retain, lab_var, color=COLORS["lab"],      lw=1.8, marker="^", ms=5, linestyle=":", alpha=0.7, label="Label MDN (Variance)")

# baseline horizontal line at 100% retain
base_acc = gauss[0]
ax.axhline(base_acc, color=COLORS["gauss"], lw=1, linestyle="--", alpha=0.4)

# annotations: peak values
peak_gauss_idx = np.argmax(gauss)
peak_res_idx   = np.argmax(res_mad)
ax.annotate(f"Gaussian peak\n{gauss[peak_gauss_idx]:.2f}%",
            xy=(retain[peak_gauss_idx], gauss[peak_gauss_idx]),
            xytext=(retain[peak_gauss_idx] + 3, gauss[peak_gauss_idx] - 1.2),
            color=COLORS["gauss"], fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", color=COLORS["gauss"], lw=1))
ax.annotate(f"Residual MDN peak\n{res_mad[peak_res_idx]:.2f}%",
            xy=(retain[peak_res_idx], res_mad[peak_res_idx]),
            xytext=(retain[peak_res_idx] + 3, res_mad[peak_res_idx] + 0.6),
            color=COLORS["res_mad"], fontsize=8, ha="left",
            arrowprops=dict(arrowstyle="->", color=COLORS["res_mad"], lw=1))

# ── axes formatting ────────────────────────────────────────────────────────────
ax.set_xlabel("Retention Rate (%)", color="#cccccc", fontsize=12)
ax.set_ylabel("Accuracy (%)", color="#cccccc", fontsize=12)
ax.set_title("Active Abstention on RewardBench\n(higher is better as uncertain samples are filtered)",
             color="white", fontsize=13, pad=12)

ax.set_xlim(52, 102)
ax.invert_xaxis()           # high retention → low retention (left → right abstention)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
ax.tick_params(colors="#aaaaaa")
for spine in ax.spines.values():
    spine.set_edgecolor("#333344")
ax.grid(color="#2a2d3a", linewidth=0.8)

legend = ax.legend(facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc",
                   fontsize=9, loc="lower left")

plt.tight_layout()

out_dir = os.path.join(os.path.dirname(__file__), "results", "plots")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "abstention_comparison.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close()
