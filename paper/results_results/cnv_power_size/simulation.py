import numpy as np
from batchdetect.mixture import HeavyMixture,parametric_bootstrap_lrt

def null_factory():
    return HeavyMixture(
                n_components=1,
                component_distribution='gennorm',
                n_init=3,
                max_iter=1000,
            )

def alt_factory():
            return HeavyMixture(
                n_components=2,
                component_distribution='gennorm',
                n_init=3,
                max_iter=1000,
            )

def compute_pvalue(x):
    res = parametric_bootstrap_lrt(
            x,  
            null_model_factory=null_factory,
            alt_model_factory=alt_factory,
            n_bootstrap=500,
            random_state=2021,
        )
    return res['p_value']


def _check_inputs(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have the same length; got {x.shape[0]} and {y.shape[0]}.")
    uniq = np.unique(y)
    if not np.array_equal(uniq, np.array([0, 1])) and not np.array_equal(uniq, np.array([0])) and not np.array_equal(uniq, np.array([1])):
        raise ValueError(f"y must contain only 0/1 labels; got unique values {uniq}.")
    return x, y


def _sample_indices(rng, idx_pool, k, replace=False):
    idx_pool = np.asarray(idx_pool)
    if k < 0:
        raise ValueError("k must be nonnegative.")
    if not replace and k > idx_pool.size:
        raise ValueError(
            f"Not enough samples to draw k={k} without replacement from pool of size {idx_pool.size}. "
            f"Set replace=True or reduce k."
        )
    return rng.choice(idx_pool, size=k, replace=replace)


def estimate_power_by_n(
    x,
    y,
    n_list,
    alpha=0.05,
    n_trials=1000,
    balanced=True,
    replace=False,
    random_state=0,
):
    """
    Estimate power: P(reject at level alpha) as a function of total sample size n.

    Assumes:
      - x is a 1D vector of measurements
      - y is a 1D vector of 0/1 group labels (true group identity)
      - compute_pvalue(x_sub, y_sub) exists and returns a scalar p-value

    Returns a dict keyed by n with:
      - power: rejection rate
      - se: standard error of rejection rate (binomial)
      - n0, n1: per-trial group sizes used
    """
    x, y = _check_inputs(x, y)
    rng = np.random.default_rng(random_state)

    idx0 = np.flatnonzero(y == 0)
    idx1 = np.flatnonzero(y == 1)

    results = {}
    for n in n_list:
        n = int(n)
        if n <= 1:
            raise ValueError("All n in n_list must be >= 2.")

        if balanced:
            n0 = n // 2
            n1 = n - n0
        else:
            # Preserve empirical proportion of y in the full data
            p1 = idx1.size / max(idx0.size + idx1.size, 1)
            n1 = int(np.round(n * p1))
            n1 = min(max(n1, 0), n)
            n0 = n - n1

        rejects = 0
        valid_trials = 0

        for _ in range(int(n_trials)):
            # If one group is absent in the dataset, power is undefined; raise early.
            if idx0.size == 0 or idx1.size == 0:
                raise ValueError("Both groups (0 and 1) must be present in y to estimate power.")

            sel0 = _sample_indices(rng, idx0, n0, replace=replace)
            sel1 = _sample_indices(rng, idx1, n1, replace=replace)

            sel = np.concatenate([sel0, sel1])
            x_sub = x[sel]
            y_sub = np.concatenate([np.zeros(n0, dtype=int), np.ones(n1, dtype=int)])

            p = compute_pvalue(x_sub)
            if np.isfinite(p):
                valid_trials += 1
                if p <= alpha:
                    rejects += 1

        if valid_trials == 0:
            power = np.nan
            se = np.nan
        else:
            power = rejects / valid_trials
            se = np.sqrt(power * (1.0 - power) / valid_trials)

        results[n] = {"power": power, "se": se, "n0": n0, "n1": n1, "valid_trials": valid_trials}

    return results


def estimate_false_positive_rate_within_group(
    x,
    y,
    n_list=(12, 20),
    alpha=0.05,
    n_trials=1000,
    replace=False,
    random_state=0,
    average_over_groups=True,
):
    """
    Estimate false positive rate under a within-group null:
      - choose samples from only one true group (all y==0 or all y==1)
      - randomly split into two pseudo-groups
      - run compute_pvalue on pseudo-labels
      - report rejection rate at level alpha

    If average_over_groups=True, computes FPR separately for true group 0 and 1,
    then averages their rejection rates (skipping any group that lacks enough samples,
    unless replace=True).

    Returns a dict keyed by n with:
      - fpr: estimated rejection rate
      - se: standard error
      - per_group: optional details if average_over_groups=True
    """
    x, y = _check_inputs(x, y)
    rng = np.random.default_rng(random_state)

    idx_by_group = {0: np.flatnonzero(y == 0), 1: np.flatnonzero(y == 1)}

    def _fpr_for_source_group(source_g, n):
        idx_pool = idx_by_group[source_g]
        if idx_pool.size == 0:
            return {"fpr": np.nan, "se": np.nan, "valid_trials": 0}

        n = int(n)
        n0 = n // 2
        n1 = n - n0

        rejects = 0
        valid_trials = 0
        pvals = []

        for _ in range(int(n_trials)):
            sel = _sample_indices(rng, idx_pool, n, replace=replace)
            # Random split into pseudo-groups
            perm = rng.permutation(n)
            a = perm[:n0]
            b = perm[n0:]

            x_sub = x[sel]
            y_sub = np.empty(n, dtype=int)
            y_sub[a] = 0
            y_sub[b] = 1

            p = compute_pvalue(x_sub)
            pvals.append(p)
            if np.isfinite(p):
                valid_trials += 1
                if p <= alpha:
                    rejects += 1

        if valid_trials == 0:
            return {"fpr": np.nan, "se": np.nan, "valid_trials": 0}

        fpr = rejects / valid_trials
        se = np.sqrt(fpr * (1.0 - fpr) / valid_trials)
        return {"fpr": fpr, "se": se, "valid_trials": valid_trials,'pvals':pvals}

    results = {}
    for n in n_list:
        n = int(n)
        if not average_over_groups:
            # Default to using group 0 as the source group
            results[n] = _fpr_for_source_group(0, n)
            continue

        per_group = {}
        fprs = []
        ses = []
        for g in (0, 1):
            out = _fpr_for_source_group(g, n)
            per_group[g] = out
            if np.isfinite(out["fpr"]):
                fprs.append(out["fpr"])
                ses.append(out["se"])

        if len(fprs) == 0:
            results[n] = {"fpr": np.nan, "se": np.nan, "per_group": per_group}
        else:
            # Simple average of per-group estimates
            fpr_avg = float(np.mean(fprs))
            # Conservative combined SE: average of SEs (simple and transparent)
            se_avg = float(np.mean([s for s in ses if np.isfinite(s)])) if ses else np.nan
            results[n] = {"fpr": fpr_avg, "se": se_avg, "per_group": per_group}

    return results


