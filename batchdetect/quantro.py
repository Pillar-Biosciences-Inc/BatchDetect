# SPDX-License-Identifier: GPL-3.0-only
#
# This file is part of BatchDetect
#
# Derived from the Bioconductor R package "quantro" (GPL-3):
#   https://bioconductor.org/packages/quantro/
#
# Copyright (C) 2014 Stephanie Hicks, Rafael Irizarry
#
# This program is free software: you may redistribute and/or modify it under
# the terms of the GNU General Public License version 3.
# This program is distributed WITHOUT ANY WARRANTY; see the GPL v3 for details.
# You should have received a copy of the GPL v3 along with this program (COPYING),
# or see https://www.gnu.org/licenses/gpl-3.0.en.html

import numpy as np
from scipy import stats


def _compute_Fnik(object_norm: np.ndarray, qRange):
    # R:
    # if(is.null(qRange)) Fnik = apply(objectNorm, 2, sort)
    # else Fnik = apply(objectNorm, 2, quantile, probs=qRange, na.rm=TRUE)
    # :contentReference[oaicite:7]{index=7}
    if qRange is None:
        return np.sort(object_norm, axis=0)
    q = np.asarray(qRange, dtype=float)

    # Use nanquantile to match quantile(..., na.rm=TRUE)
    # Keep method/interpolation aligned with R's default type=7 (linear).
    try:
        return np.nanquantile(object_norm, q=q, axis=0, method="linear")
    except TypeError:
        return np.nanquantile(object_norm, q=q, axis=0, interpolation="linear")


def quantro(
    object,
    groupFactor,
    B: int = 0,
    qRange=None,
    useMedianNormalized: bool = True,
    verbose: bool = True,
    seed=None,
    n_jobs: int = 1,
):
    X = np.asarray(object, dtype=float)
    if X.ndim != 2:
        raise ValueError("object must be 2D (n_features, n_samples).")

    group = np.asarray(groupFactor)
    if X.shape[1] != group.shape[0]:
        raise ValueError(
            "Number of columns in object does not match length of groupFactor."
        )

    # R checks B==0 message, then B<0 stop later :contentReference[oaicite:8]{index=8}
    if B < 0:
        raise ValueError("Must pick B greater than or equal to 0.")

    nT = X.shape[1]
    group_levels, group_codes = np.unique(group, return_inverse=True)
    K = int(group_levels.shape[0])
    nk = np.bincount(group_codes, minlength=K).astype(int)

    # Median step must NOT ignore NaNs for parity:
    # objectMedians <- apply(object, 2, stats::median) :contentReference[oaicite:9]{index=9}
    object_medians = np.median(X, axis=0)
    object_medians = np.round(
        object_medians, 7
    )  # :contentReference[oaicite:10]{index=10}

    # ANOVA logic parity (structure); F/p computation method may differ slightly,
    # but NA behavior is now consistent with R medians.
    anova_res = None
    if np.unique(object_medians).size == 1:
        if verbose:
            # message text mirrors R :contentReference[oaicite:11]{index=11}
            print(
                "[quantro] All median values equal. No ANOVA performed. No median normalization."
            )
    else:
        # R: anova(lm(objectMedians ~ groupFactor)) :contentReference[oaicite:12]{index=12}
        # lm drops NA rows; mimic that by dropping NaN medians.
        valid = ~np.isnan(object_medians)
        med_v = object_medians[valid]
        codes_v = group_codes[valid]

        try:
            groups_for_anova = [med_v[codes_v == k] for k in range(K)]
            groups_for_anova = [g for g in groups_for_anova if g.size > 0]
            if len(groups_for_anova) >= 2:
                fstat, pval = stats.f_oneway(*groups_for_anova)
                anova_res = {"F": float(fstat), "pvalue": float(pval)}
                if verbose:
                    if pval < 0.05:
                        print(
                            "[quantro] Average medians of the distributions are not equal across groups."
                        )
                    else:
                        print(
                            "[quantro] Average medians of the distributions are equal across groups."
                        )
        except Exception:
            anova_res = None

    # Median normalization is performed if useMedianNormalized==TRUE
    # even if the message says "No median normalization." (that is also how R behaves)
    # :contentReference[oaicite:13]{index=13} :contentReference[oaicite:14]{index=14}
    if useMedianNormalized:
        X_norm = X - object_medians[None, :]
    else:
        X_norm = X

    if verbose:
        print("[quantro] Calculating the quantro test statistic.")

    Fnik = _compute_Fnik(X_norm, qRange=qRange)

    # Must NOT use nanmean here; R uses rowMeans/colMeans without na.rm :contentReference[oaicite:15]{index=15}
    Fndotdot = np.mean(Fnik, axis=1)
    Fndotk = np.stack(
        [np.mean(Fnik[:, group_codes == k], axis=1) for k in range(K)], axis=1
    )

    betweenDiff = np.mean((Fndotk - Fndotdot[:, None]) ** 2, axis=0)
    MSb = float(
        np.sum(betweenDiff * nk) / (K - 1)
    )  # :contentReference[oaicite:16]{index=16}

    within_blocks = []
    for k in range(K):
        cols = np.where(group_codes == k)[0]
        block = Fnik[:, cols] - np.mean(Fnik[:, cols], axis=1)[:, None]
        within_blocks.append(block)

    withinDiff = np.mean(np.concatenate(within_blocks, axis=1) ** 2, axis=0)
    MSe = float(
        np.sum(withinDiff) / (nT - K)
    )  # :contentReference[oaicite:17]{index=17}

    if MSe == 0:
        if MSb == 0:
            quantroStat = np.nan
        else:
            quantroStat = np.inf
    else:
        quantroStat = float(MSb / MSe)

    if B == 0:
        if verbose:
            print(
                "[quantro] No permutation testing performed. Use B > 0 for permutation testing."
            )
        return {
            "anova": anova_res,
            "MSbetween": MSb,
            "MSwithin": MSe,
            "quantroStat": quantroStat,
            "quantroStatPerm": None,
            "quantroPvalPerm": None,
        }

    # Optional parity: R permutation code likely errors if any nk == 1 :contentReference[oaicite:18]{index=18}
    # Uncomment if you want that behavior.
    # if np.any(nk == 1):
    #     raise ValueError("R quantro permutation code fails with singleton groups (nk==1).")

    if verbose:
        print("[quantro] Starting permutation testing.")

    rng = np.random.default_rng(seed)
    perms = np.stack([rng.permutation(group_codes) for _ in range(B)], axis=0)

    def one_test(code_vec: np.ndarray) -> float:
        Fndotk_perm = np.stack(
            [np.mean(Fnik[:, code_vec == k], axis=1) for k in range(K)], axis=1
        )
        betweenDiff_perm = np.mean(
            (Fndotk_perm - Fndotdot[:, None]) ** 2, axis=0
        )
        MSb_perm = float(np.sum(betweenDiff_perm * nk) / (K - 1))

        blocks = []
        for k in range(K):
            cols = np.where(code_vec == k)[0]
            blocks.append(
                Fnik[:, cols] - np.mean(Fnik[:, cols], axis=1)[:, None]
            )
        withinDiff_perm = np.mean(np.concatenate(blocks, axis=1) ** 2, axis=0)
        MSe_perm = float(np.sum(withinDiff_perm) / (nT - K))

        return float(MSb_perm / MSe_perm)

    F_perm = np.asarray([one_test(perms[i]) for i in range(B)], dtype=float)

    # Parity with R's mean(F_perm > quantroStat):
    # if any NA involved, mean(...) becomes NA. :contentReference[oaicite:19]{index=19}
    if np.isnan(quantroStat) or np.any(np.isnan(F_perm)):
        p_perm = np.nan
    else:
        p_perm = float(np.mean(F_perm > quantroStat))

    return {
        "anova": anova_res,
        "MSbetween": MSb,
        "MSwithin": MSe,
        "quantroStat": quantroStat,
        "quantroStatPerm": F_perm,
        "quantroPvalPerm": p_perm,
    }
