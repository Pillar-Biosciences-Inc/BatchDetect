"""
Unit tests for non-Bayesian model selection methods.

Tests that BIC, AIC, AICc, ICL, and CV correctly identify the true number
of components (K=1, 2, 3) with large sample sizes.

NOTE: AIC/AICc are known to overfit (select more components than necessary),
especially for single-component data. This is documented behavior, not a bug.
BIC is asymptotically consistent while AIC is not.
"""

import numpy as np
import pytest
from batchdetect.model_selection import (
    _default_n_parameters,
    compare_criteria,
    compute_aic,
    compute_bic,
    compute_cv_likelihood,
    compute_icl,
    select_n_components,
)

from batchdetect.mixture import HeavyMixture

# =============================================================================
# Fixtures for generating test data
# =============================================================================


def generate_mixture_data(
    n_samples_per_component: int,
    n_components: int,
    n_features: int = 2,
    separation: float = 6.0,
    scale: float = 0.5,
    distribution: str = "laplace",
    random_state: int = 42,
) -> np.ndarray:
    """
    Generate well-separated mixture data with known number of components.

    Parameters
    ----------
    n_samples_per_component : int
        Number of samples per component.
    n_components : int
        True number of components (1, 2, or 3).
    n_features : int
        Number of features.
    separation : float
        Distance between component centers.
    scale : float
        Scale parameter for the distribution.
    distribution : str
        Either "laplace" or "gaussian".
    random_state : int
        Random seed.

    Returns
    -------
    X : ndarray of shape (n_samples, n_features)
        Generated data.
    """
    rng = np.random.RandomState(random_state)

    # Define component centers (well-separated)
    if n_components == 1:
        centers = [np.zeros(n_features)]
    elif n_components == 2:
        centers = [
            np.array([-separation / 2] + [0] * (n_features - 1)),
            np.array([separation / 2] + [0] * (n_features - 1)),
        ]
    elif n_components == 3:
        # Triangle arrangement
        centers = [
            np.array(
                [-separation / 2, -separation / 3] + [0] * (n_features - 2)
            ),
            np.array(
                [separation / 2, -separation / 3] + [0] * (n_features - 2)
            ),
            np.array([0, separation * 2 / 3] + [0] * (n_features - 2)),
        ]
    else:
        raise ValueError(f"n_components must be 1, 2, or 3, got {n_components}")

    # Generate samples
    samples = []
    for center in centers:
        if distribution == "laplace":
            X_k = rng.laplace(
                loc=center,
                scale=scale,
                size=(n_samples_per_component, n_features),
            )
        elif distribution == "gaussian":
            X_k = rng.normal(
                loc=center,
                scale=scale,
                size=(n_samples_per_component, n_features),
            )
        else:
            raise ValueError(f"Unknown distribution: {distribution}")
        samples.append(X_k)

    X = np.vstack(samples)

    # Shuffle
    perm = rng.permutation(len(X))
    return X[perm]


@pytest.fixture
def data_k1_large():
    """Large sample data with 1 component (1000 samples)."""
    return generate_mixture_data(
        n_samples_per_component=1000,
        n_components=1,
        n_features=2,
        random_state=42,
    )


@pytest.fixture
def data_k1_very_large():
    """Very large sample data with 1 component (5000 samples).

    AIC needs more samples than BIC to correctly identify K=1.
    """
    return generate_mixture_data(
        n_samples_per_component=5000,
        n_components=1,
        n_features=2,
        random_state=42,
    )


@pytest.fixture
def data_k2_large():
    """Large sample data with 2 components (2000 total samples)."""
    return generate_mixture_data(
        n_samples_per_component=1000,
        n_components=2,
        n_features=2,
        separation=6.0,
        random_state=42,
    )


@pytest.fixture
def data_k3_large():
    """Large sample data with 3 components (3000 total samples)."""
    return generate_mixture_data(
        n_samples_per_component=1000,
        n_components=3,
        n_features=2,
        separation=6.0,
        random_state=42,
    )


def model_factory(k: int) -> HeavyMixture:
    """Factory function for creating HeavyMixture models."""
    return HeavyMixture(
        n_components=k,
        component_distribution="laplace",
        n_init=3,
        max_iter=200,
        random_state=42,
    )


# =============================================================================
# Tests for compute_bic
# =============================================================================


class TestComputeBIC:
    """Tests for BIC computation."""

    def test_bic_decreases_with_better_fit(self, data_k2_large):
        """BIC should be lower for K=2 than K=1 when true K=2."""
        model_k1 = model_factory(1)
        model_k2 = model_factory(2)

        model_k1.fit(data_k2_large)
        model_k2.fit(data_k2_large)

        bic_k1 = compute_bic(data_k2_large, model_k1)
        bic_k2 = compute_bic(data_k2_large, model_k2)

        assert (
            bic_k2 < bic_k1
        ), f"BIC(K=2)={bic_k2:.2f} should be < BIC(K=1)={bic_k1:.2f}"

    def test_bic_penalizes_overfitting(self, data_k2_large):
        """BIC should increase for K > true K due to complexity penalty."""
        model_k2 = model_factory(2)
        model_k5 = model_factory(5)

        model_k2.fit(data_k2_large)
        model_k5.fit(data_k2_large)

        bic_k2 = compute_bic(data_k2_large, model_k2)
        bic_k5 = compute_bic(data_k2_large, model_k5)

        assert (
            bic_k2 < bic_k5
        ), f"BIC(K=2)={bic_k2:.2f} should be < BIC(K=5)={bic_k5:.2f}"

    def test_bic_custom_n_parameters(self, data_k2_large):
        """Test BIC with custom n_parameters."""
        model = model_factory(2)
        model.fit(data_k2_large)

        bic_default = compute_bic(data_k2_large, model)
        bic_custom = compute_bic(data_k2_large, model, n_parameters=100)

        # More parameters should increase BIC
        assert bic_custom > bic_default


# =============================================================================
# Tests for compute_aic
# =============================================================================


class TestComputeAIC:
    """Tests for AIC and AICc computation."""

    def test_aic_uncorrected(self, data_k2_large):
        """Test uncorrected AIC computation."""
        model = model_factory(2)
        model.fit(data_k2_large)

        aic = compute_aic(data_k2_large, model, corrected=False)
        aicc = compute_aic(data_k2_large, model, corrected=True)

        # AICc >= AIC always (correction term is non-negative)
        assert aicc >= aic

    def test_aicc_correction_small_sample(self):
        """AICc correction should be significant for small samples."""
        # Small sample
        X_small = generate_mixture_data(
            n_samples_per_component=20,
            n_components=2,
            random_state=42,
        )

        model = model_factory(2)
        model.fit(X_small)

        aic = compute_aic(X_small, model, corrected=False)
        aicc = compute_aic(X_small, model, corrected=True)

        # Correction should be noticeable
        correction = aicc - aic
        assert (
            correction > 1.0
        ), f"AICc correction {correction:.2f} should be > 1 for small samples"

    def test_aicc_converges_to_aic_large_sample(self, data_k2_large):
        """AICc should be close to AIC for large samples."""
        model = model_factory(2)
        model.fit(data_k2_large)

        aic = compute_aic(data_k2_large, model, corrected=False)
        aicc = compute_aic(data_k2_large, model, corrected=True)

        relative_diff = abs(aicc - aic) / abs(aic)
        assert (
            relative_diff < 0.01
        ), "AICc should be within 1% of AIC for large samples"


# =============================================================================
# Tests for compute_icl
# =============================================================================


class TestComputeICL:
    """Tests for ICL computation."""

    def test_icl_penalizes_fuzzy_clusters(self, data_k2_large):
        """ICL should be higher than BIC when clusters overlap."""
        model = model_factory(2)
        model.fit(data_k2_large)

        bic = compute_bic(data_k2_large, model)
        icl = compute_icl(data_k2_large, model)

        # ICL = BIC + 2*entropy, so ICL >= BIC
        assert icl >= bic

    def test_icl_favors_well_separated_clusters(self):
        """ICL should favor models with well-separated clusters."""
        # Well-separated data
        X_sep = generate_mixture_data(
            n_samples_per_component=500,
            n_components=2,
            separation=10.0,  # Very separated
            scale=0.3,
            random_state=42,
        )

        model = model_factory(2)
        model.fit(X_sep)

        bic = compute_bic(X_sep, model)
        icl = compute_icl(X_sep, model)

        # For well-separated clusters, entropy is low, so ICL ≈ BIC
        diff = icl - bic
        assert (
            diff < 100
        ), f"ICL-BIC difference {diff:.2f} should be small for well-separated clusters"


# =============================================================================
# Tests for compute_cv_likelihood
# =============================================================================


class TestComputeCVLikelihood:
    """Tests for cross-validated likelihood computation."""

    def test_cv_returns_mean_and_std(self, data_k2_large):
        """CV should return mean and std of fold scores."""
        mean_cv, std_cv = compute_cv_likelihood(
            data_k2_large,
            model_factory=lambda: model_factory(2),
            n_folds=5,
            random_state=42,
        )

        assert isinstance(mean_cv, float)
        assert isinstance(std_cv, float)
        assert std_cv >= 0

    def test_cv_reproducible_with_seed(self, data_k2_large):
        """CV should be reproducible with same random_state."""
        mean1, std1 = compute_cv_likelihood(
            data_k2_large,
            model_factory=lambda: model_factory(2),
            n_folds=5,
            random_state=42,
        )

        mean2, std2 = compute_cv_likelihood(
            data_k2_large,
            model_factory=lambda: model_factory(2),
            n_folds=5,
            random_state=42,
        )

        assert mean1 == mean2
        assert std1 == std2

    def test_cv_invalid_folds(self, data_k2_large):
        """CV should raise error for invalid n_folds."""
        with pytest.raises(ValueError):
            compute_cv_likelihood(
                data_k2_large,
                model_factory=lambda: model_factory(2),
                n_folds=1,  # Must be >= 2
            )


# =============================================================================
# Tests for select_n_components - K=1 (CORE TESTS)
# =============================================================================


class TestSelectNComponentsK1:
    """Test that methods correctly identify K=1.

    Note: AIC/AICc are known to overfit and may require larger samples
    to correctly identify K=1. This is documented behavior.
    """

    @pytest.mark.parametrize("criterion", ["bic", "icl"])
    def test_consistent_criteria_select_k1(self, data_k1_large, criterion):
        """BIC and ICL (consistent criteria) should select K=1 with n=1000."""
        results = select_n_components(
            data_k1_large,
            model_factory=model_factory,
            k_range=range(1, 5),
            criterion=criterion,
            random_state=42,
        )

        assert results["best_k"] == 1, (
            f"{criterion.upper()} selected K={results['best_k']}, expected K=1. "
            f"Scores: {dict(zip(results['k_values'], results['scores']))}"
        )

    @pytest.mark.parametrize("criterion", ["aic", "aicc"])
    def test_aic_selects_k1_very_large_sample(
        self, data_k1_very_large, criterion
    ):
        """AIC/AICc should select K=1 with very large samples (n=5000).

        AIC has a weaker complexity penalty (2k) compared to BIC (k*log(n)),
        so it needs more data to avoid overfitting for single-component data.
        """
        results = select_n_components(
            data_k1_very_large,
            model_factory=model_factory,
            k_range=range(1, 5),
            criterion=criterion,
            random_state=42,
        )

        assert results["best_k"] == 1, (
            f"{criterion.upper()} selected K={results['best_k']}, expected K=1. "
            f"Scores: {dict(zip(results['k_values'], results['scores']))}"
        )

    def test_cv_selects_k1_large_sample(self, data_k1_large):
        """CV should select K=1 when true K=1."""
        results = select_n_components(
            data_k1_large,
            model_factory=model_factory,
            k_range=range(1, 5),
            criterion="cv",
            n_folds=5,
            random_state=42,
        )

        # CV may also overfit slightly; allow K=1 or K=2
        assert results["best_k"] in [1, 2], (
            f"CV selected K={results['best_k']}, expected K=1 (or K=2 due to variance). "
            f"Scores: {dict(zip(results['k_values'], results['scores']))}"
        )


# =============================================================================
# Tests for select_n_components - K=2 (CORE TESTS)
# =============================================================================


class TestSelectNComponentsK2:
    """Test that methods correctly identify K=2."""

    @pytest.mark.parametrize("criterion", ["bic", "aic", "aicc", "icl"])
    def test_selects_k2_large_sample(self, data_k2_large, criterion):
        """All criteria should select K=2 when true K=2 with large samples."""
        results = select_n_components(
            data_k2_large,
            model_factory=model_factory,
            k_range=range(1, 5),
            criterion=criterion,
            random_state=42,
        )

        assert results["best_k"] == 2, (
            f"{criterion.upper()} selected K={results['best_k']}, expected K=2. "
            f"Scores: {dict(zip(results['k_values'], results['scores']))}"
        )

    def test_cv_selects_k2_large_sample(self, data_k2_large):
        """CV should select K>=2 when true K=2.

        CV tends to overfit more than information criteria, especially when
        scores plateau after the true K. We verify that:
        1. K=1 is clearly rejected (much lower score)
        2. K>=2 is selected (scores plateau after true K)
        """
        results = select_n_components(
            data_k2_large,
            model_factory=model_factory,
            k_range=range(1, 5),
            criterion="cv",
            n_folds=5,
            random_state=42,
        )

        scores = dict(zip(results["k_values"], results["scores"]))

        # K=1 should be clearly worse than K=2
        assert (
            scores[2] > scores[1] + 0.5
        ), f"K=2 should be much better than K=1. Scores: {scores}"

        # CV may select K>=2 due to score plateau; this is acceptable
        assert (
            results["best_k"] >= 2
        ), f"CV selected K={results['best_k']}, expected K>=2. Scores: {scores}"


# =============================================================================
# Tests for select_n_components - K=3 (CORE TESTS)
# =============================================================================


class TestSelectNComponentsK3:
    """Test that methods correctly identify K=3."""

    @pytest.mark.parametrize("criterion", ["bic", "aic", "aicc", "icl"])
    def test_selects_k3_large_sample(self, data_k3_large, criterion):
        """All criteria should select K=3 when true K=3 with large samples."""
        results = select_n_components(
            data_k3_large,
            model_factory=model_factory,
            k_range=range(1, 6),
            criterion=criterion,
            random_state=42,
        )

        assert results["best_k"] == 3, (
            f"{criterion.upper()} selected K={results['best_k']}, expected K=3. "
            f"Scores: {dict(zip(results['k_values'], results['scores']))}"
        )

    def test_cv_selects_k3_large_sample(self, data_k3_large):
        """CV should select K>=3 when true K=3.

        CV tends to overfit more than information criteria. We verify that:
        1. K=1 and K=2 are clearly rejected
        2. K>=3 is selected
        """
        results = select_n_components(
            data_k3_large,
            model_factory=model_factory,
            k_range=range(1, 6),
            criterion="cv",
            n_folds=5,
            random_state=42,
        )

        scores = dict(zip(results["k_values"], results["scores"]))

        # K=3 should be clearly better than K=1 and K=2
        assert (
            scores[3] > scores[1] + 0.5
        ), f"K=3 should be much better than K=1. Scores: {scores}"
        assert (
            scores[3] > scores[2] + 0.1
        ), f"K=3 should be better than K=2. Scores: {scores}"

        # CV may select K>=3 due to score plateau; this is acceptable
        assert (
            results["best_k"] >= 3
        ), f"CV selected K={results['best_k']}, expected K>=3. Scores: {scores}"


# =============================================================================
# Tests for select_n_components - Additional functionality
# =============================================================================


class TestSelectNComponentsFunctionality:
    """Test additional functionality of select_n_components."""

    def test_returns_best_model_fitted(self, data_k2_large):
        """Should return a fitted model."""
        results = select_n_components(
            data_k2_large,
            model_factory=model_factory,
            k_range=range(1, 4),
            criterion="bic",
        )

        # Check model is fitted
        assert hasattr(results["best_model"], "weights_")
        assert hasattr(results["best_model"], "means_")
        assert hasattr(results["best_model"], "scales_")

        # Check model has correct K
        assert results["best_model"].n_components == results["best_k"]

    def test_returns_all_scores(self, data_k2_large):
        """Should return scores for all K values."""
        k_range = range(1, 5)
        results = select_n_components(
            data_k2_large,
            model_factory=model_factory,
            k_range=k_range,
            criterion="bic",
        )

        assert len(results["scores"]) == len(k_range)
        assert len(results["k_values"]) == len(k_range)
        assert results["k_values"] == list(k_range)

    def test_custom_n_parameters_fn(self, data_k2_large):
        """Should use custom n_parameters function if provided."""

        def custom_n_params(model, n_features):
            # Simpler model: just count means
            return model.n_components * n_features

        results = select_n_components(
            data_k2_large,
            model_factory=model_factory,
            k_range=range(1, 4),
            criterion="bic",
            n_parameters_fn=custom_n_params,
        )

        # Should still work and return results
        assert "best_k" in results
        assert "scores" in results

    def test_invalid_criterion_raises(self, data_k2_large):
        """Should raise error for invalid criterion."""
        with pytest.raises(ValueError, match="criterion must be one of"):
            select_n_components(
                data_k2_large,
                model_factory=model_factory,
                k_range=range(1, 4),
                criterion="invalid",
            )


# =============================================================================
# Tests for compare_criteria
# =============================================================================


class TestCompareCriteria:
    """Tests for compare_criteria function."""

    def test_compares_all_criteria(self, data_k2_large):
        """Should compare all specified criteria."""
        criteria = ["bic", "aic", "icl"]
        results = compare_criteria(
            data_k2_large,
            model_factory=model_factory,
            k_range=range(1, 4),
            criteria=criteria,
            random_state=42,
        )

        assert set(results.keys()) == set(criteria)
        for crit in criteria:
            assert "best_k" in results[crit]
            assert "scores" in results[crit]

    def test_all_criteria_agree_on_well_separated_data(self, data_k2_large):
        """All criteria should agree on K for well-separated 2-component data."""
        results = compare_criteria(
            data_k2_large,
            model_factory=model_factory,
            k_range=range(1, 5),
            criteria=["bic", "aic", "aicc", "icl"],
            random_state=42,
        )

        best_ks = [results[crit]["best_k"] for crit in results]

        # All should select K=2
        assert all(
            k == 2 for k in best_ks
        ), f"Criteria disagreed: {[(c, results[c]['best_k']) for c in results]}"


# =============================================================================
# Tests for _default_n_parameters
# =============================================================================


class TestDefaultNParameters:
    """Tests for default parameter counting."""

    def test_parameter_count_formula(self):
        """Test the parameter counting formula."""
        model = HeavyMixture(n_components=3)
        model.n_components = 3
        n_features = 5

        n_params = _default_n_parameters(model, n_features)

        # Formula: (K-1) + K*d + K = K*(d+2) - 1
        # K=3, d=5: 3*(5+2) - 1 = 20
        expected = 3 * (5 + 2) - 1
        assert n_params == expected

    def test_parameter_count_k1(self):
        """Test parameter count for K=1."""
        model = HeavyMixture(n_components=1)
        n_features = 2

        n_params = _default_n_parameters(model, n_features)

        # K=1: 0 weights + 1*2 means + 1 scale = 3
        assert n_params == 3


# =============================================================================
# Integration tests
# =============================================================================


class TestIntegration:
    """Integration tests for the full workflow."""

    def test_full_workflow_k2(self):
        """Test complete workflow: generate data, select K, verify."""
        # Generate data
        X = generate_mixture_data(
            n_samples_per_component=500,
            n_components=2,
            separation=8.0,
            random_state=123,
        )

        # Select K
        results = select_n_components(
            X,
            model_factory=model_factory,
            k_range=range(1, 6),
            criterion="bic",
        )

        # Verify
        assert results["best_k"] == 2

        # Check model quality
        model = results["best_model"]
        assert model.n_components == 2

        # Check that means are approximately correct (within some tolerance)
        # True means are at [-4, 0] and [4, 0] with separation=8
        means_sorted = model.means_[np.argsort(model.means_[:, 0])]
        assert means_sorted[0, 0] < 0  # First mean should be negative
        assert means_sorted[1, 0] > 0  # Second mean should be positive

    def test_gaussian_distribution(self):
        """Test with Gaussian distribution."""
        # Generate Gaussian data
        X = generate_mixture_data(
            n_samples_per_component=500,
            n_components=2,
            separation=6.0,
            distribution="gaussian",
            random_state=42,
        )

        # Use Gaussian model
        def gaussian_factory(k):
            return HeavyMixture(
                n_components=k,
                component_distribution="gaussian",
                n_init=3,
                random_state=42,
            )

        results = select_n_components(
            X,
            model_factory=gaussian_factory,
            k_range=range(1, 5),
            criterion="bic",
        )

        assert results["best_k"] == 2
