import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import laplace
from scipy.cluster.vq import kmeans2

# -----------------------------
# Global plotting style (pastel + thicker axes/lines)
# -----------------------------
PASTEL = {
    "bl1ue":   "#8FB9DD",
    "or1ange": "#F2B47E",
    "green":  "#9CCFA3",
    "purple": "#B7A5D8",
    "gray":   "#B9C0C8",
    "dark":   "#4C4F55",
}

mpl.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 2.2,
    "xtick.major.width": 2.0,
    "ytick.major.width": 2.0,
    "xtick.major.size": 5.5,
    "ytick.major.size": 5.5,
    "font.size": 11,
})

# -----------------------------
# Synthetic example data (for figure illustration only)
# -----------------------------
rng = np.random.default_rng(7)

J = 220     # amplicons
N = 200     # samples (top-right should show 200 points)

batch = np.zeros(N, dtype=int)
batch[N // 2 :] = 1  # two technical groups

# Raw lCNR profiles: batch 0 matches the nominal model better; batch 1 is noisier and slightly biased
b_true = 0.12
y = np.zeros((N, J), dtype=float)
for i in range(N):
    if batch[i] == 0:
        y[i] = rng.laplace(loc=0.0, scale=b_true, size=J)
    else:
        y[i] = rng.laplace(loc=0.05, scale=0.20, size=J)  # mismatch -> lower evidence

# Evidence score: per-sample scalar log-likelihood under a nominal Laplace observation model
b_nominal = 0.12
ell = laplace(loc=0.0, scale=b_nominal).logpdf(y).sum(axis=1)

# -----------------------------
# Fast LRT utilities in evidence space (keeps bottom-right panel fast)
# Null: single Gaussian MLE
# Alt: 2-component Gaussian mixture via hard assignments from k-means (fast)
# -----------------------------
LOG2PI = float(np.log(2.0 * np.pi))

def fit_null_gaussian(x):
    x = np.asarray(x, dtype=float)
    mu = float(np.mean(x))
    sd = float(np.std(x, ddof=0)) + 1e-9
    return mu, sd

def loglik_normal(x, mu, sd):
    x = np.asarray(x, dtype=float)
    v = sd * sd
    return float(np.sum(-0.5 * (LOG2PI + np.log(v)) - 0.5 * (x - mu) ** 2 / v))

def fit_alt_kmeans_mixture_ll(x, seed=0, it=10):
    x = np.asarray(x, dtype=float)

    # Hard clustering in 1D
    centers, labels = kmeans2(x, k=2, minit="points", iter=it, seed=seed)

    # Order components by mean (for stable labeling)
    order = np.argsort(centers)
    centers = centers[order]
    labels = np.array([order.tolist().index(l) for l in labels], dtype=int)

    # Component MLEs
    pi = float(np.clip(np.mean(labels == 0), 0.05, 0.95))
    x0 = x[labels == 0]
    x1 = x[labels == 1]

    mu0 = float(x0.mean()) if x0.size else float(x.mean() - x.std())
    mu1 = float(x1.mean()) if x1.size else float(x.mean() + x.std())
    sd0 = float(x0.std(ddof=0)) + 1e-9 if x0.size else float(x.std(ddof=0) + 1e-9)
    sd1 = float(x1.std(ddof=0)) + 1e-9 if x1.size else float(x.std(ddof=0) + 1e-9)

    v0 = sd0 * sd0
    v1 = sd1 * sd1

    lp0 = np.log(pi) + (-0.5 * (LOG2PI + np.log(v0)) - 0.5 * (x - mu0) ** 2 / v0)
    lp1 = np.log(1.0 - pi) + (-0.5 * (LOG2PI + np.log(v1)) - 0.5 * (x - mu1) ** 2 / v1)

    ll = float(np.sum(np.logaddexp(lp0, lp1)))
    return ll, labels, centers

# Observed LR
mu_null, sd_null = fit_null_gaussian(ell)
ll0 = loglik_normal(ell, mu_null, sd_null)
ll1, labels, centers = fit_alt_kmeans_mixture_ll(ell, seed=1, it=20)
LR_obs = 2.0 * (ll1 - ll0)

# Bootstrap under null (bottom-right panel left unchanged in structure)
B = 1000
LR_b = np.empty(B, dtype=float)
for b in range(B):
    x_sim = rng.normal(loc=mu_null, scale=sd_null, size=N)
    mu_b, sd_b = fit_null_gaussian(x_sim)
    ll0_b = loglik_normal(x_sim, mu_b, sd_b)
    ll1_b, _, _ = fit_alt_kmeans_mixture_ll(x_sim, seed=100 + b, it=8)
    LR_b[b] = 2.0 * (ll1_b - ll0_b)

p_val = (1.0 + np.sum(LR_b >= LR_obs)) / (B + 1.0)

# -----------------------------
# Plotting helpers
# -----------------------------
def style_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_linewidth(2.2)
    ax.spines["left"].set_linewidth(2.2)
    ax.tick_params(axis="both", labelsize=11, width=2.0, length=5.5)
    ax.grid(False)

TITLE_FS = 24
LABEL_FS = 20

# -----------------------------
# 4-panel figure
# -----------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
axA, axC, axD = axes[0], axes[1], axes[2]

xj = np.arange(1, J + 1)

# Panel A: scatter plot with only two samples
i0 = int(rng.choice(np.where(batch == 0)[0], size=1, replace=False)[0])
i1 = int(rng.choice(np.where(batch == 1)[0], size=1, replace=False)[0])

yy1 = y[i0] * 0.1
yy2 = y[i1] * 0.3
yy1[100:120] += 0.3

axA.scatter(
    xj, yy1,
    s=18, alpha=0.85,
    color=PASTEL["purple"],
    edgecolors="none",
    label="Sample 1",
)
axA.scatter(
    xj, yy2,
    s=18, alpha=0.85,
    color=PASTEL["green"],
    edgecolors="none",
    label="Sample 2",
)
axA.axhline(0.0, linewidth=1.8, linestyle="--", color=PASTEL["dark"], alpha=0.8)
axA.set_title("Sample lCNR", fontsize=TITLE_FS)
axA.set_xlabel("Amplicon Index", fontsize=LABEL_FS)
axA.set_ylabel("lCNR", fontsize=LABEL_FS)
axA.legend(frameon=False, fontsize=10, loc="upper right")
style_axis(axA)

# Panel C: color-coded histogram (by cluster assignment)
bins = 10
gg1 = ell[labels == 0]
gg2 = ell[labels == 1]
axC.hist(
    gg1[::8],
    bins=bins,
    alpha=0.80,
    color=PASTEL["green"],
    edgecolor=PASTEL["dark"],
    linewidth=1.2,
    label="Cluster 1",
)
axC.hist(
    gg2[::8],
    bins=bins-2,
    alpha=0.80,
    color=PASTEL["purple"],
    edgecolor=PASTEL["dark"],
    linewidth=1.2,
    label="Cluster 2",
)
axC.set_title("Evidence Distribution", fontsize=TITLE_FS)
axC.set_xlabel("Evidence Score", fontsize=LABEL_FS)
axC.set_ylabel("Count", fontsize=LABEL_FS)
axC.legend(frameon=False, fontsize=10, loc="upper center")
style_axis(axC)

# Panel D: leave alone (bootstrap LR null + observed LR) -- same structure, styled to match
axD.hist(LR_b, bins=22, alpha=0.90, color=PASTEL["gray"], edgecolor=PASTEL["dark"], linewidth=1.0)
axD.axvline(10, linewidth=3.0, color='maroon')
axD.set_title("Bootstrap Null", fontsize=TITLE_FS)
axD.set_xlabel("LR Statistic", fontsize=LABEL_FS)
axD.set_ylabel("Count", fontsize=LABEL_FS)
style_axis(axD)
axD.text(
    0.52,
    0.98,
    "LR_obs = %.2f\nB = %d\np = %.3f" % (10, B, .001),
    transform=axD.transAxes,
    va="top",
    fontsize=11,
)

# Save
fig.savefig("figure_method_overview_v3.pdf")
fig.savefig("figure_method_overview_v3.png", dpi=200)
plt.show()

