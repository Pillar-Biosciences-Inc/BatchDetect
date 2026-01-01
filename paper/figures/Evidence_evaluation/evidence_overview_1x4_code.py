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

# =============================================================================
# 1x4 FIGURE: Evidence Computation Overview
# =============================================================================

fig, axes = plt.subplots(1, 4, figsize=(12, 2.8))

color_prior = '#4878A8'
color_posterior = '#C44E52'
color_intermediate = '#888888'
color_main = '#4878A8'
color_accent = '#C44E52'

# -----------------------------------------------------------------------------
# Plot 1: Tempered posteriors between N(0,1) prior and N(1, 0.1) posterior
# -----------------------------------------------------------------------------
ax = axes[0]

prior_mu, prior_sigma = -1, .5
post_mu, post_sigma = 1, 0.1
taus = [0, 0.25, 0.5, 0.75, 1.0]
x = np.linspace(-3, 2, 500)

for i, tau in enumerate(taus):
    # Tempered posterior: prior^(1-τ) * posterior^τ
    # For Gaussians, this gives another Gaussian with interpolated parameters
    prec_prior = (1 - tau) / prior_sigma**2
    prec_post = tau / post_sigma**2
    
    if prec_prior + prec_post > 0:
        sigma_tau = 1 / np.sqrt(prec_prior + prec_post)
        mu_tau = sigma_tau**2 * (prec_prior * prior_mu + prec_post * post_mu)
    else:
        mu_tau, sigma_tau = prior_mu, prior_sigma
    
    pdf = stats.norm.pdf(x, mu_tau, sigma_tau)
    
    if tau == 0:
        ax.plot(x, pdf, color=color_prior, linewidth=2, label='Prior')
    elif tau == 1:
        ax.plot(x, pdf, color=color_posterior, linewidth=2, label='Posterior')
    else:
        ax.plot(x, pdf, color=color_intermediate, linewidth=1, alpha=0.5 + 0.2*tau)

ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Density')
ax.set_title('Tempered Posteriors', loc='left', fontweight='bold')
ax.legend(loc='upper left', frameon=False, fontsize=8)
ax.set_xlim(-3, 2)
ax.text(-0.3, 0.35, r'$\tau=0$', fontsize=7, color=color_prior)
ax.text(1.15, 3.5, r'$\tau=1$', fontsize=7, color=color_posterior)

# -----------------------------------------------------------------------------
# Plot 2: Thermodynamic Integration
# -----------------------------------------------------------------------------
ax = axes[1]

n_temps = 50
tau = np.linspace(0, 1, n_temps)

np.random.seed(123)
base_curve = -150 + 120 * (1 - np.exp(-3 * tau))
noise = np.random.normal(0, 3, n_temps)
noise[0] = noise[-1] = 0
expected_ll = base_curve + noise * np.sqrt(tau * (1 - tau) + 0.01)

ax.plot(tau, expected_ll, 'o-', color=color_main, markersize=3, linewidth=1.5)
ax.fill_between(tau, expected_ll.min() - 10, expected_ll, alpha=0.3, color=color_main)

#ax.annotate('', xy=(0.5, -75), xytext=(0.5, -115),
#            arrowprops=dict(arrowstyle='->', color='gray', lw=1.2))
ax.text(0.53, -95, r'$\log p(y)$', fontsize=14, va='center')
ax.axvline(x=0, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)
ax.axvline(x=1, color='gray', linestyle=':', alpha=0.7, linewidth=0.8)

ax.set_xlabel(r'Temperature $\tau$')
ax.set_ylabel(r'$\mathbb{E}[\log p(y|\theta)]$')
ax.set_title('Thermodynamic Integration', loc='left', fontweight='bold')
ax.set_xlim(-0.05, 1.05)

# -----------------------------------------------------------------------------
# Plot 3: Importance samples (x = value, y = weight)
# -----------------------------------------------------------------------------
ax = axes[2]

np.random.seed(42)
n_samples = 10

samples = np.random.normal(0, 1, n_samples)
target_mu, target_sigma = 1, 0.3
proposal_mu, proposal_sigma = 0, 1

log_weights = (stats.norm.logpdf(samples, target_mu, target_sigma) - 
               stats.norm.logpdf(samples, proposal_mu, proposal_sigma))
weights = np.exp(log_weights - log_weights.max())
weights = weights / weights.sum()

markerline, stemlines, baseline = ax.stem(samples, weights, linefmt='-', 
                                           markerfmt='o', basefmt=' ')
plt.setp(stemlines, color=color_main, linewidth=1.5)
plt.setp(markerline, color=color_main, markersize=8)
ax.axhline(y=0, color='gray', linewidth=0.5)

ax.set_xlabel(r'Sample value $\theta^{(i)}$')
ax.set_ylabel(r'Weight $w^{(i)}$')
ax.set_title('Importance weights', loc='left', fontweight='bold')
ax.set_ylim(-0.02, weights.max() * 1.15)

# -----------------------------------------------------------------------------
# Plot 4: Cumulative log evidence (SMC)
# -----------------------------------------------------------------------------
ax = axes[3]

np.random.seed(789)
n_timesteps = 30
n_particles = 50

true_state = np.zeros(n_timesteps)
for t in range(1, n_timesteps):
    if t == 15:
        true_state[t] = true_state[t-1] + 0.4
    else:
        true_state[t] = true_state[t-1] + np.random.normal(0, 0.02)

observations = true_state + np.random.laplace(0, 0.08, n_timesteps)

particles = np.zeros((n_timesteps, n_particles))
cumulative_log_evidence = np.zeros(n_timesteps)

particles[0, :] = np.random.normal(0, 0.3, n_particles)
log_weights_0 = -np.abs(observations[0] - particles[0, :]) / 0.08
max_lw = log_weights_0.max()
mean_weight = np.exp(log_weights_0 - max_lw).mean() * np.exp(max_lw)
cumulative_log_evidence[0] = np.log(mean_weight)

weights_normalized = np.exp(log_weights_0 - max_lw)
weights_normalized = weights_normalized / weights_normalized.sum()
indices = np.random.choice(n_particles, n_particles, p=weights_normalized)
particles[0, :] = particles[0, indices]

for t in range(1, n_timesteps):
    jump_mask = np.random.random(n_particles) < 0.03
    particles[t, :] = particles[t-1, :].copy()
    particles[t, jump_mask] += np.random.normal(0, 0.3, jump_mask.sum())
    particles[t, ~jump_mask] += np.random.normal(0, 0.02, (~jump_mask).sum())
    
    log_weights = -np.abs(observations[t] - particles[t, :]) / 0.08
    max_lw = log_weights.max()
    mean_weight = np.exp(log_weights - max_lw).mean() * np.exp(max_lw)
    cumulative_log_evidence[t] = cumulative_log_evidence[t-1] + np.log(mean_weight)
    
    weights_normalized = np.exp(log_weights - max_lw)
    weights_normalized = weights_normalized / weights_normalized.sum()
    indices = np.random.choice(n_particles, n_particles, p=weights_normalized)
    particles[t, :] = particles[t, indices]

time = np.arange(n_timesteps)
ax.plot(time, cumulative_log_evidence, color=color_main, linewidth=2)
ax.fill_between(time, cumulative_log_evidence.min() - 5, cumulative_log_evidence, 
                alpha=0.3, color=color_main)

ax.axhline(y=cumulative_log_evidence[-1], color=color_accent, linestyle='--', 
           linewidth=1, alpha=0.7)
ax.text(n_timesteps * 0.95, cumulative_log_evidence[-1] + 2, 
        r'$\log p(y)$', fontsize=8, ha='right', color=color_accent)

ax.set_xlabel('Time step $t$')
ax.set_ylabel(r'$\log p(y_{1:t})$')
ax.set_title('SMC Evidence', loc='left', fontweight='bold')
ax.set_xlim(0, n_timesteps - 1)

plt.tight_layout()
plt.savefig('evidence_computation_overview.pdf', format='pdf', bbox_inches='tight')
plt.savefig('evidence_computation_overview.png', format='png', dpi=300, bbox_inches='tight')
plt.show()
