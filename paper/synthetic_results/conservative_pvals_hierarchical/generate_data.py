import numpy as np
import pickle
from tqdm import trange
from scipy.stats import invgamma, norm

def sample_softlaplace(mu, tau, size, rng):
    """
    Sample from SoftLaplace(mu, tau) using inverse CDF.
    pdf(x) = (1/(4*tau)) * sech^2((x - mu)/(2*tau))
    """
    U = rng.random(size)
    V = 2.0 * U - 1.0
    return mu + 2.0 * tau * np.arctanh(V)

def hierarchical_model_sample(
    lengths,
    alpha_sigma=5.1,
    beta_sigma=1.1,
    alpha_tau0=20.1,
    beta_tau0=2.1,
    alpha_tau=20.1,
    beta_tau=2.1,
    seed=None,
):
    """
    Same as before, but now RNG is controlled by a seed.
    """
    rng = np.random.default_rng(seed)
    J = len(lengths)

    # Hyperpriors
    mu0 = rng.normal(loc=0.0, scale=0.1)

    sigma2 = invgamma.rvs(a=alpha_sigma, scale=beta_sigma, random_state=rng)
    sigma = np.sqrt(sigma2)

    tau0_2 = invgamma.rvs(a=alpha_tau0, scale=beta_tau0, random_state=rng)
    tau0 = np.sqrt(tau0_2)

    eta = rng.normal(loc=0.0, scale=1.0, size=J)
    z2 = invgamma.rvs(a=alpha_tau, scale=beta_tau, size=J, random_state=rng)

    mu_j = mu0 + sigma * eta
    tau2_j = tau0_2 * z2
    tau_j = np.sqrt(tau2_j)

    X_list = []
    for j, n in enumerate(lengths):
        x = sample_softlaplace(mu_j[j], tau_j[j], size=n, rng=rng)
        X_list.append(x)

    return {
        "mu0": mu0,
        "sigma": sigma,
        "tau0": tau0,
        "mu_j": mu_j,
        "tau_j": tau_j,
        "X_list": X_list,
    }
lengths = [14, 12, 12, 15, 11, 7, 10, 12 , 5, 14 , 327]

n_reps = 200
n_batch = 20

myDict = {}

for i in trange(n_reps):
    Xl = []
    for j in range(n_batch):
        out = hierarchical_model_sample(lengths,
                    seed=i*1000+j)
        Xl.append(out['X_list'])
    myDict[i] = Xl
    with open('Core_synthetic.p','wb') as f:
        pickle.dump(myDict,f)


