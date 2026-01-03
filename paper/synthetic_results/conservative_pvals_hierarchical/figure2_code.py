import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


np.random.seed(42)
n_simulations = 200
true_a = 1.35  # Conservative test
p_values = stats.beta.rvs(true_a, 1, size=n_simulations)

# =============================================================================
# STATISTICAL ANALYSIS
# =============================================================================
def fit_beta_a1_posterior(p_values, alpha_prior=1.0, lambda_prior=1.0):
    """
    Fit Beta(a, 1) model with Gamma(alpha, lambda) prior on 'a'.
    Posterior: Gamma(alpha + n, lambda - sum(log(p)))
    """
    n = len(p_values)
    sum_log_p = np.sum(np.log(p_values))
    alpha_post = alpha_prior + n
    lambda_post = lambda_prior - sum_log_p
    
    a_mean = alpha_post / lambda_post
    a_lower = stats.gamma.ppf(0.025, alpha_post, scale=1/lambda_post)
    a_upper = stats.gamma.ppf(0.975, alpha_post, scale=1/lambda_post)
    
    return a_mean, a_lower, a_upper

def clopper_pearson_upper(k, n, alpha=0.05):
    """Upper bound of Clopper-Pearson confidence interval."""
    if k == n:
        return 1.0
    return stats.beta.ppf(1 - alpha/2, k + 1, n - k)

# Fit model
a_est, a_lower, a_upper = fit_beta_a1_posterior(p_values)
n_below_05 = np.sum(p_values < 0.05)
cp_upper = clopper_pearson_upper(n_below_05, len(p_values))

print(f"Beta(a, 1) posterior: a = {a_est:.3f} (95% CI: [{a_lower:.3f}, {a_upper:.3f}])")
print(f"Conservative (a > 1): {a_lower > 1}")
print(f"Type I error at α=0.05: {n_below_05}/{len(p_values)} = {n_below_05/len(p_values):.3f}")
print(f"Clopper-Pearson 97.5% upper bound: {cp_upper:.3f}")

# =============================================================================
# FIGURE CREATION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))

# Color scheme
color_hist = '#4878A8'
color_uniform = '#2D2D2D'
color_beta = '#C44E52'
color_ci = '#AAAAAA'

# Panel A: Histogram with density overlays
ax = axes[0]
ax.hist(p_values, bins=15, density=True, alpha=0.7, color=color_hist, 
        edgecolor='white', linewidth=0.8, label='Observed')
ax.axhline(y=1, color=color_uniform, linestyle='--', linewidth=1.5, 
           label=r'Uniform$(0,1)$')

x_dense = np.linspace(0.001, 0.999, 500)
ax.plot(x_dense, stats.beta.pdf(x_dense, a_est, 1), color=color_beta, 
        linewidth=2, label=fr'Beta$(\hat{{a}}, 1)$, $\hat{{a}}={a_est:.2f}$')
ax.fill_between(x_dense, stats.beta.pdf(x_dense, a_lower, 1), 
                stats.beta.pdf(x_dense, a_upper, 1), 
                color=color_beta, alpha=0.15, label='95% CI')

ax.axvline(x=0.05, color='gray', linestyle=':', linewidth=1, alpha=0.8)
ax.text(0.07, ax.get_ylim()[1]*0.9, r'$\alpha=0.05$', fontsize=8, color='gray')
ax.set_xlabel(r'$p$-value')
ax.set_ylabel('Density')
ax.set_xlim(0, 1)
ax.legend(loc='upper right', frameon=False)
ax.set_title('(A) P-value distribution', loc='left', fontweight='bold')

# Panel B: Q-Q Plot
ax = axes[1]
p_sorted = np.sort(p_values)
n = len(p_values)
theoretical_quantiles = (np.arange(1, n + 1) - 0.5) / n

# 95% CI bands from order statistics
ci_lower = stats.beta.ppf(0.025, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)
ci_upper = stats.beta.ppf(0.975, np.arange(1, n + 1), n - np.arange(1, n + 1) + 1)

ax.fill_between(theoretical_quantiles, ci_lower, ci_upper, 
                color=color_ci, alpha=0.4, label='95% CI')
ax.plot([0, 1], [0, 1], color=color_uniform, linestyle='--', linewidth=1.5, 
        label='Uniform reference')
ax.scatter(theoretical_quantiles, p_sorted, s=12, color=color_hist, 
           alpha=0.7, edgecolors='none', label='Observed')
ax.plot(theoretical_quantiles, stats.beta.ppf(theoretical_quantiles, a_est, 1), 
        color=color_beta, linewidth=1.5, label=fr'Beta$({a_est:.2f}, 1)$ expected')

ax.set_xlabel('Theoretical quantiles (Uniform)')
ax.set_ylabel('Observed quantiles')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect('equal')
ax.legend(loc='lower right', frameon=False, fontsize=8)
ax.set_title('(B) Q-Q plot', loc='left', fontweight='bold')

plt.tight_layout()
plt.savefig('figure2_pvalue_calibration.pdf', format='pdf', bbox_inches='tight')
plt.savefig('figure2_pvalue_calibration.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
