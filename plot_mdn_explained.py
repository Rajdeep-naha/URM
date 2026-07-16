"""
Visual explanation of MDN(Variance) vs MDN(MAD) as uncertainty measures.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import logistic as logistic_dist

out = "/home/f20221218/.gemini/antigravity-ide/brain/16ede877-14ac-4ab7-beec-d4d786001061/plots/mdn_uncertainty_explained.png"

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.patch.set_facecolor("#0f1117")
for ax in axes:
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="#aaaaaa")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333344")
    ax.grid(color="#2a2d3a", linewidth=0.8, alpha=0.5)

x = np.linspace(-8, 8, 1000)

# ─── Panel 1: Logistic Mixture (3 components) ───────────────────────────────
ax = axes[0]
pi  = np.array([0.5, 0.3, 0.2])
mu  = np.array([-1.0, 1.5, 3.5])
s   = np.array([0.6, 1.0, 0.5])
COLS = ["#64b5f6", "#a5d6a7", "#ef9a9a"]

total_pdf = np.zeros_like(x)
for i in range(3):
    pdf_i = logistic_dist.pdf(x, loc=mu[i], scale=s[i])
    ax.fill_between(x, pi[i]*pdf_i, alpha=0.25, color=COLS[i])
    ax.plot(x, pi[i]*pdf_i, color=COLS[i], lw=1.5, linestyle="--",
            label=f"Component {i+1}: π={pi[i]}, μ={mu[i]}, s={s[i]}")
    total_pdf += pi[i]*pdf_i

ax.plot(x, total_pdf, color="white", lw=2.5, label="Mixture PDF")
ax.set_title("MDN: Mixture of 3 Logistics\n(K=3 components per attribute)", color="white", fontsize=10)
ax.set_xlabel("Residual value", color="#cccccc")
ax.set_ylabel("p(x)", color="#cccccc")
ax.legend(fontsize=7, facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc", loc="upper left")

# ─── Panel 2: Variance vs MAD shown on a NARROW (certain) mixture ───────────
ax = axes[1]
ax.set_title("Certain sample\n→ Low spread = Low Uncertainty", color="white", fontsize=10)

pi2 = np.array([0.1, 0.8, 0.1])
mu2 = np.array([-0.5, 0.0, 0.5])
s2  = np.array([0.3, 0.4, 0.3])

total2 = sum(pi2[i]*logistic_dist.pdf(x, loc=mu2[i], scale=s2[i]) for i in range(3))
ax.fill_between(x, total2, alpha=0.3, color="#a5d6a7")
ax.plot(x, total2, color="#a5d6a7", lw=2.5)

mu_mix2 = np.sum(pi2 * mu2)
var2 = np.sum(pi2 * ((np.pi**2/3) * s2**2 + (mu2 - mu_mix2)**2))
std2 = np.sqrt(var2)
mad2 = np.sum(pi2 * (2*s2*np.log1p(np.exp(mu2/s2)) - mu2))  # MAD_0 approx

ax.axvline(mu_mix2, color="white", lw=1.5, linestyle="-", label=f"Mean = {mu_mix2:.2f}")
ax.axvspan(mu_mix2 - std2, mu_mix2 + std2, alpha=0.15, color="#64b5f6",
           label=f"±SD = {std2:.2f}  [Variance={var2:.2f}]")
ax.axvspan(mu_mix2 - mad2, mu_mix2 + mad2, alpha=0.0)
ax.annotate("", xy=(mu_mix2 + mad2, 0.55), xytext=(mu_mix2, 0.55),
            arrowprops=dict(arrowstyle="<->", color="#ffd54f", lw=2))
ax.text(mu_mix2 + mad2/2, 0.60, f"MAD={mad2:.2f}", ha="center", color="#ffd54f", fontsize=8)

ax.set_xlabel("Residual value", color="#cccccc")
ax.set_xlim(-4, 4)
ax.legend(fontsize=7.5, facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc")
ax.text(0.5, -0.15, "→ Both Variance & MAD are SMALL\n→ Sample kept (not abstained)",
        ha="center", transform=ax.transAxes, color="#a5d6a7", fontsize=8)

# ─── Panel 3: Variance vs MAD on WIDE (uncertain) mixture ───────────────────
ax = axes[2]
ax.set_title("Uncertain sample\n→ Wide spread = High Uncertainty", color="white", fontsize=10)

pi3 = np.array([0.4, 0.2, 0.4])
mu3 = np.array([-2.5, 0.0, 2.5])
s3  = np.array([1.0, 0.8, 1.0])

total3 = sum(pi3[i]*logistic_dist.pdf(x, loc=mu3[i], scale=s3[i]) for i in range(3))
ax.fill_between(x, total3, alpha=0.3, color="#ef9a9a")
ax.plot(x, total3, color="#ef9a9a", lw=2.5)

mu_mix3 = np.sum(pi3 * mu3)
var3 = np.sum(pi3 * ((np.pi**2/3) * s3**2 + (mu3 - mu_mix3)**2))
std3 = np.sqrt(var3)
mad3 = np.sum(pi3 * (2*s3*np.log1p(np.exp(mu3/s3)) - mu3))

ax.axvline(mu_mix3, color="white", lw=1.5, linestyle="-", label=f"Mean = {mu_mix3:.2f}")
ax.axvspan(mu_mix3 - std3, mu_mix3 + std3, alpha=0.15, color="#64b5f6",
           label=f"±SD = {std3:.2f}  [Variance={var3:.2f}]")
ax.annotate("", xy=(mu_mix3 + mad3, 0.12), xytext=(mu_mix3, 0.12),
            arrowprops=dict(arrowstyle="<->", color="#ffd54f", lw=2))
ax.text(mu_mix3 + mad3/2, 0.145, f"MAD={mad3:.2f}", ha="center", color="#ffd54f", fontsize=8)

ax.set_xlabel("Residual value", color="#cccccc")
ax.set_xlim(-8, 8)
ax.legend(fontsize=7.5, facecolor="#1a1d27", edgecolor="#333344", labelcolor="#cccccc")
ax.text(0.5, -0.15, "→ Both Variance & MAD are LARGE\n→ Sample abstained (filtered out)",
        ha="center", transform=ax.transAxes, color="#ef9a9a", fontsize=8)

fig.suptitle("MDN Uncertainty: Variance vs MAD\n"
             "Both measure spread of the residual distribution — large spread = uncertain prediction",
             color="white", fontsize=12, y=1.02)

plt.tight_layout()
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out}")
plt.close()
