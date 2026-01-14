"""Figure 2 (bootstrap LRT) creation script.

This script is a cleaned, command-line friendly conversion of the notebook
`Figure2Creation_LMR.ipynb` (adapted to parametric bootstrap LRT).

What it does
- Loads per-batch log-likelihood normalizing constants (logZs) from one or more
  pickle files containing dictionaries: {batch_key: logZs}.
- Computes parametric bootstrap LRT p-values for each batch using HeavyMixture (K=1 vs K=2).
- Optionally saves p-values to a text file.
- Fits a Beta(a, 1) model to the p-values under a Gamma prior on `a`.
- Produces a two-panel PDF:
    (A) Histogram of p-values + Uniform(0,1) reference + fitted Beta(a,1) density
    (B) Q-Q plot with order-statistic 95% CI band + fitted Beta(a,1) quantiles

Notes
- The original notebook used two pickle files and wrote the second set into an
  offset slice of a preallocated array. Here we instead concatenate results from
  all provided pickle files in a deterministic order (sorted batch keys per file).
- File paths are interpreted relative to the current working directory unless
  you pass absolute paths.

Example
  python Figure2Creation_LMR.py \
      --pickle Likelihoods.p --pickle LogLikelihoods2.p \
      --pvals-out Computed_pvals2_lmr.txt \
      --figure-out Figure2_synthetic_lmr.pdf

"""

from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

from batchdetect.mixture import HeavyMixture,parametric_bootstrap_lrt  # type: ignore
from batchdetect.pvalue_evaluation import alpha_posterior_credible_interval



LOGGER = logging.getLogger(__name__)


def _sample_from_model(model: Any, n_samples: int, seed: int) -> np.ndarray:
    """Sample from a fitted model with best-effort seeding support."""
    try:
        Xb = model.sample(n_samples, random_state=seed)
    except TypeError:
        try:
            Xb = model.sample(n_samples, seed=seed)
        except TypeError:
            Xb = model.sample(n_samples)

    # Some APIs return (X, labels)
    if isinstance(Xb, tuple):
        Xb = Xb[0]
    return np.asarray(Xb)


def _heavymixture_factory(
    n_components: int,
    component_distribution: str,
    n_init: int,
    max_iter: int,
) -> Any:
    """Create a HeavyMixture instance, passing a seed if supported."""
    kwargs = dict(
        n_components=int(n_components),
        component_distribution=str(component_distribution),
        n_init=int(n_init),
        max_iter=int(max_iter),
    )
    try:
        return HeavyMixture(**kwargs)
    except TypeError:
        return HeavyMixture(**kwargs)


def compute_bootstrap_pvalues(
    batch_dict: Dict,
    n_bootstrap: int = 500,
    random_state: Union[int, np.random.Generator, None] = None,
    component_distribution: str = "gaussian",
    n_init: int = 3,
    max_iter: int = 1000,
) -> Tuple[np.ndarray, List[str]]:
    """Compute parametric bootstrap LRT p-values for each batch entry in `batch_dict`.

    Each batch is tested as:
      - H0: 1-component mixture (K=1)
      - H1: 2-component mixture (K=2)

    The data passed to the mixture model is the per-sample logZ vector for that batch.
    """
    keys = sorted(batch_dict.keys(), key=lambda x: str(x))
    pvals: List[float] = []

    rng = np.random.default_rng(random_state)

    for k in tqdm(keys, desc="Computing bootstrap LRT p-values", unit="batch"):
        X = np.asarray(batch_dict[k])

        def null_factory() -> Any:
            return _heavymixture_factory(
                n_components=1,
                component_distribution=component_distribution,
                n_init=n_init,
                max_iter=max_iter,
            )

        def alt_factory() -> Any:
            return _heavymixture_factory(
                n_components=2,
                component_distribution=component_distribution,
                n_init=n_init,
                max_iter=max_iter,
            )

        batch_seed = int(rng.integers(0, 2**32 - 1))
        res = parametric_bootstrap_lrt(
            X,
            null_model_factory=null_factory,
            alt_model_factory=alt_factory,
            n_bootstrap=int(n_bootstrap),
            random_state=batch_seed,
        )
        pvals.append(float(res["p_value"]))

    return np.asarray(pvals, dtype=float), [str(k) for k in keys]


def load_pickle_dict(path: Path) -> Dict:
    """Load a pickle file expected to contain a dict mapping keys to arrays."""
    with path.open("rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, dict):
        raise TypeError(f"Expected dict in {path}, got {type(obj)}")
    return obj


def compute_lmr_pvalues(*args: Any, **kwargs: Any) -> Tuple[np.ndarray, List[str]]:
    """Deprecated: retained for backward compatibility.

    The original notebook/script computed p-values using an LMR test on HeavyMixture.
    This script now computes p-values using a parametric bootstrap LRT (K=1 vs K=2)
    implemented in `compute_bootstrap_pvalues`.
    """
    return compute_bootstrap_pvalues(*args, **kwargs)


def clip_pvalues_open_interval(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map p-values into (0, 1) to avoid log/ppf issues."""
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def fit_beta_a1_posterior(p_values: np.ndarray, alpha_prior: float = 1.0, lambda_prior: float = 1.0) -> Tuple[float, float, float]:
    """Fit Beta(a, 1) model with Gamma(alpha, lambda) prior on 'a'.

    Posterior: Gamma(alpha + n, lambda - sum(log(p))).
    Returns posterior mean and 95% central credible interval.
    """
    p_values = np.asarray(p_values, dtype=float)
    n = p_values.size
    sum_log_p = float(np.sum(np.log(p_values)))
    alpha_post = alpha_prior + n
    lambda_post = lambda_prior - sum_log_p

    a_mean = alpha_post / lambda_post
    a_lower = stats.gamma.ppf(0.025, alpha_post, scale=1.0 / lambda_post)
    a_upper = stats.gamma.ppf(0.975, alpha_post, scale=1.0 / lambda_post)
    return float(a_mean), float(a_lower), float(a_upper)


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Upper bound of Clopper-Pearson confidence interval for a binomial proportion."""
    if k >= n:
        return 1.0
    return float(stats.beta.ppf(1.0 - alpha / 2.0, k + 1, n - k))


def stylize_axes(ax: plt.Axes) -> None:
    """Apply consistent aesthetics."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    ax.tick_params(width=1.2, length=4)
    ax.title.set_fontsize(20)
    ax.xaxis.label.set_fontsize(16)
    ax.yaxis.label.set_fontsize(16)


def make_figure(p_values: np.ndarray, a_est: float, a_lower: float, a_upper: float, figure_out: Path) -> None:
    """Create and save the Figure 2 PDF."""

    # Pastel-inspired palette from the notebook.
    color_hist = "#88AADD"     # softer blue
    color_uniform = "#666666"  # dark gray
    color_beta = "#E07B91"     # muted pink
    color_ci = "#CCCCCC"       # light gray

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

    # Panel A: Histogram and density
    ax = axes[0]
    ax.hist(
        p_values,
        bins=15,
        density=True,
        alpha=0.7,
        color=color_hist,
        edgecolor="white",
        linewidth=0.8,
        label="Observed",
    )

    ax.axhline(
        y=1.0,
        color=color_uniform,
        linestyle="--",
        linewidth=1.5,
        label=r"Uniform$(0,1)$",
    )

    x_dense = np.linspace(0.001, 0.999, 500)
    ax.plot(
        x_dense,
        stats.beta.pdf(x_dense, a_est, 1),
        color=color_beta,
        linewidth=2,
        label=fr"Beta$(\hat{{a}}, 1)$, $\hat{{a}}={a_est:.2f}$",
    )

    ax.fill_between(
        x_dense,
        stats.beta.pdf(x_dense, a_lower, 1),
        stats.beta.pdf(x_dense, a_upper, 1),
        color=color_beta,
        alpha=0.15,
        label="95% CI",
    )

    ax.set_xlabel("P-values")
    ax.set_ylabel("Density")
    ax.set_title("Bootstrap LRT P-Values")
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, fontsize=14)
    stylize_axes(ax)

    # Panel B: Q-Q Plot
    ax = axes[1]
    p_sorted = np.sort(p_values)
    n = p_sorted.size
    theoretical_quantiles = (np.arange(1, n + 1) - 0.5) / n

    # 95% CI bands from order statistics under Uniform(0,1)
    ci_lower = stats.beta.ppf(0.025, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)
    ci_upper = stats.beta.ppf(0.975, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)

    ax.fill_between(
        theoretical_quantiles,
        ci_lower,
        ci_upper,
        color=color_ci,
        alpha=0.4,
        label="95% CI",
    )

    ax.plot([0, 1], [0, 1], color=color_uniform, linestyle="--", linewidth=1.5)

    ax.scatter(
        theoretical_quantiles,
        p_sorted,
        s=16,
        color=color_hist,
        alpha=0.7,
        edgecolors="none",
        label="Observed",
    )

    ax.plot(
        theoretical_quantiles,
        stats.beta.ppf(theoretical_quantiles, a_est, 1),
        color=color_beta,
        linewidth=1.5,
        label=fr"Beta$({a_est:.2f}, 1)$",
    )

    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Observed quantiles")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", frameon=False, fontsize=14)
    ax.set_title("Bootstrap LRT Q-Q Plot", fontsize=20)
    stylize_axes(ax)

    plt.tight_layout()
    figure_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_out, bbox_inches="tight")
    LOGGER.info("Saved figure to %s", figure_out)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create Figure 2 (bootstrap LRT) from likelihood pickles."
    )
    p.add_argument(
        "--pickle",
        dest="pickles",
        action="append",
        required=True,
        help="Path to a pickle file containing a dict {batch_key: logZs}. "
        "Repeat to include multiple files (results are concatenated).",
    )
    p.add_argument(
        "--pvals-out",
        type=str,
        default="Computed_pvals2_lmr.txt",
        help="Output path for p-values (one per line).",
    )
    p.add_argument(
        "--figure-out",
        type=str,
        default="Figure2_synthetic_lmr.pdf",
        help="Output path for the figure PDF.",
    )
    p.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Number of parametric bootstrap replicates for the LRT null distribution.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed controlling bootstrap and model initialization.",
    )
    p.add_argument(
        "--component-distribution",
        type=str,
        default="gaussian",
        help="Component distribution family for HeavyMixture (e.g., gaussian, student_t, gennorm, hypsecant).",
    )
    p.add_argument(
        "--n-init",
        type=int,
        default=3,
        help="Number of random initializations for HeavyMixture fitting.",
    )
    p.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum number of EM iterations for HeavyMixture fitting.",
    )
    p.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Clipping epsilon to keep p-values in (0,1).",
    )
    p.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return p.parse_args(argv)



def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    all_pvals: List[np.ndarray] = []

    rng = np.random.default_rng(int(args.seed))

    for pkl in args.pickles:
        path = Path(pkl)
        if not path.exists():
            raise FileNotFoundError(f"Pickle not found: {path}")
        LOGGER.info("Loading %s", path)
        d = load_pickle_dict(path)
        pvals, _keys = compute_bootstrap_pvalues(
            d,
            n_bootstrap=int(args.n_bootstrap),
            random_state=int(rng.integers(0, 2**32 - 1)),
            component_distribution=str(args.component_distribution),
            n_init=int(args.n_init),
            max_iter=int(args.max_iter),
        )
        all_pvals.append(pvals)

    p_values = np.concatenate(all_pvals) if all_pvals else np.array([], dtype=float)
    if p_values.size == 0:
        raise RuntimeError("No p-values computed. Check input pickle(s).")

    p_values = clip_pvalues_open_interval(p_values, eps=args.eps)

    # Optional summary statistics / bounds, mirroring the notebook's calculations.
    a_est, a_lower, a_upper = fit_beta_a1_posterior(p_values)
    n_below_05 = int(np.sum(p_values < 0.05))
    cp_upper = clopper_pearson_upper(n_below_05, int(p_values.size))
    LOGGER.info("Posterior mean a_hat = %.4f (95%% CI: %.4f, %.4f)", a_est, a_lower, a_upper)
    LOGGER.info("Count(p < 0.05) = %d / %d; Clopper-Pearson upper = %.4f", n_below_05, p_values.size, cp_upper)

    if alpha_posterior_credible_interval is not None:
        try:
            alpha_ci = alpha_posterior_credible_interval(p_values, 2, 2, 0.05)
            LOGGER.info("alpha_posterior_credible_interval(...) = %s", str(alpha_ci))
        except Exception as e:  # pragma: no cover
            LOGGER.warning("alpha_posterior_credible_interval failed: %s", e)

    # Save p-values (one per line), matching the notebook's output.
    pvals_out = Path(args.pvals_out)
    pvals_out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(pvals_out, p_values, fmt="%0.8f", delimiter="\n")
    LOGGER.info("Saved p-values to %s", pvals_out)

    figure_out = Path(args.figure_out)
    make_figure(p_values, a_est, a_lower, a_upper, figure_out)

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
