import numpy as np
import pytest

from batchdetect.pvalue_evaluation import (
    alpha_posterior_credible_interval,
    conservativeness_bound,
)


def test_alpha_posterior_valid_inputs():
    """Test alpha_posterior_credible_interval with valid inputs."""
    x_samples = [0.1, 0.2, 0.3, 0.4, 0.5]
    prior_shape = 1.0
    prior_rate = 1.0
    alpha = 0.05

    lower, upper = alpha_posterior_credible_interval(
        x_samples, prior_shape, prior_rate, alpha
    )

    assert isinstance(lower, float)
    assert isinstance(upper, float)
    assert lower < upper
    assert lower > 0


def test_alpha_posterior_edge_case_single_sample():
    """Test alpha_posterior_credible_interval with a single sample."""
    x_samples = [0.5]
    prior_shape = 1.0
    prior_rate = 1.0
    alpha = 0.05

    lower, upper = alpha_posterior_credible_interval(
        x_samples, prior_shape, prior_rate, alpha
    )

    assert lower < upper


def test_alpha_posterior_invalid_x_samples_empty():
    """Test alpha_posterior_credible_interval with empty x_samples."""
    with pytest.raises(
        ValueError, match="x_samples must contain at least one value"
    ):
        alpha_posterior_credible_interval([], 1.0, 1.0, 0.05)


def test_alpha_posterior_invalid_x_samples_out_of_bounds():
    """Test alpha_posterior_credible_interval with x_samples <= 0 or >= 1."""
    with pytest.raises(
        ValueError, match="All x_samples must be in the open interval"
    ):
        alpha_posterior_credible_interval([0.0, 0.5], 1.0, 1.0, 0.05)

    with pytest.raises(
        ValueError, match="All x_samples must be in the open interval"
    ):
        alpha_posterior_credible_interval([0.5, 1.0], 1.0, 1.0, 0.05)

    with pytest.raises(
        ValueError, match="All x_samples must be in the open interval"
    ):
        alpha_posterior_credible_interval([-0.1, 0.5], 1.0, 1.0, 0.05)


def test_alpha_posterior_invalid_prior_params():
    """Test alpha_posterior_credible_interval with invalid prior parameters."""
    x_samples = [0.5]
    with pytest.raises(
        ValueError, match="prior_shape and prior_rate must be positive"
    ):
        alpha_posterior_credible_interval(x_samples, 0.0, 1.0, 0.05)

    with pytest.raises(
        ValueError, match="prior_shape and prior_rate must be positive"
    ):
        alpha_posterior_credible_interval(x_samples, 1.0, -1.0, 0.05)


def test_alpha_posterior_invalid_alpha():
    """Test alpha_posterior_credible_interval with invalid alpha."""
    x_samples = [0.5]
    with pytest.raises(ValueError, match="alpha must be in"):
        alpha_posterior_credible_interval(x_samples, 1.0, 1.0, 0.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        alpha_posterior_credible_interval(x_samples, 1.0, 1.0, 1.0)

    with pytest.raises(ValueError, match="alpha must be in"):
        alpha_posterior_credible_interval(x_samples, 1.0, 1.0, 1.1)


def test_alpha_posterior_rate_check():
    """
    Test that ValueError is raised if posterior rate becomes non-positive.
    """
    x_samples = [1e-10]
    prior_shape = 1.0
    prior_rate = 1.0
    alpha = 0.05
    # Should not raise
    alpha_posterior_credible_interval(x_samples, prior_shape, prior_rate, alpha)


def test_conservativeness_bound_valid_inputs_structure():
    """Test conservativeness_bound return structure with valid inputs."""
    null_pvals = [0.01, 0.04, 0.1, 0.5, 0.8]
    alpha = 0.05
    gamma = 0.05

    result = conservativeness_bound(null_pvals, alpha, gamma)

    assert isinstance(result, dict)
    assert "m" in result
    assert "X" in result
    assert "theta_hat" in result
    assert "theta_upper" in result
    assert "is_conservative" in result

    assert result["m"] == 5
    assert result["X"] == 2  # 0.01 and 0.04 are <= 0.05
    assert result["theta_hat"] == 0.4
    assert isinstance(result["is_conservative"], (bool, np.bool_))


def test_conservativeness_bound_all_significant():
    """Test conservativeness_bound when all p-values are <= alpha."""
    null_pvals = [0.01, 0.02, 0.03]
    alpha = 0.05
    result = conservativeness_bound(null_pvals, alpha)

    assert result["X"] == 3
    assert result["theta_hat"] == 1.0
    assert result["theta_upper"] == 1.0


def test_conservativeness_bound_none_significant():
    """Test conservativeness_bound when no p-values are <= alpha."""
    null_pvals = [0.1, 0.2, 0.3]
    alpha = 0.05
    result = conservativeness_bound(null_pvals, alpha)

    assert result["X"] == 0
    assert result["theta_hat"] == 0.0
    assert (
        result["theta_upper"] > 0.0
    )  # Upper bound of 0 should be > 0 for finite samples


def test_conservativeness_bound_empty_pvals():
    """Test conservativeness_bound with empty null_pvals."""
    with pytest.raises(
        ValueError, match="null_pvals must contain at least one value"
    ):
        conservativeness_bound([])


def test_conservativeness_bound_is_conservative_logic():
    """Test conservativeness_bound is_conservative flag logic."""
    null_pvals = [0.5] * 100
    alpha = 0.05
    result = conservativeness_bound(null_pvals, alpha)

    assert result["X"] == 0
    assert result["theta_hat"] == 0.0
    # Check if it turned out conservative as expected
    if result["theta_upper"] <= alpha:
        assert result["is_conservative"]
    else:
        assert not result["is_conservative"]


def test_conservativeness_bound_numpy_input():
    """Test conservativeness_bound with numpy array input."""
    null_pvals = np.array([0.01, 0.1])
    result = conservativeness_bound(null_pvals)
    assert result["m"] == 2
