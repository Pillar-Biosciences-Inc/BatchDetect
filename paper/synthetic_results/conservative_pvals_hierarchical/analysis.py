import numpy as np
import matplotlib.pyplot as plt
from batchdetect.mixture import HeavyMixture,parametric_bootstrap_lrt
from batchdetect.pvalue_evaluation import alpha_posterior_credible_interval
from typing import Callable, Any, Dict
from tqdm import  trange
import pickle
from batchdetect.lmr import lo_mendell_rubin_lrt,weighted_chisq_lrt_num_components

with open('LogLikelihoods.p','rb') as f:
    likelihood_dict = pickle.load(f)


def null_factory(seed: int) -> HeavyMixture:
    return HeavyMixture(
        n_components=1,
        component_distribution='gennorm',
        n_init=3,
        max_iter=1000,
    )

def alt_factory(seed: int) -> HeavyMixture:
    return HeavyMixture(
        n_components=2,
        component_distribution='gennorm',
        n_init=3,
        max_iter=1000,
    )

## Actually run test
keys = likelihood_dict.keys()

n = len(keys)
p_values_bootstrap = np.zeros(n)
p_values_lmr = np.zeros(n)
p_values_w_chisq = np.zeros(n)

for i,key in enumerate(keys):
    if i % 2 == 0:
        print(i)
    x = likelihood_dict[key]
    res = parametric_bootstrap_lrt(x,
                                    null_factory,alt_factory,
                                    1000,random_state=2021)

    res_lmr = lo_mendell_rubin_lrt(x,k_null=1,k_alt=2)
    res_wc = weighted_chisq_lrt_num_components(x,1,2)

    p_values_bootstrap[i] = res['p_value']
    p_values_lmr[i] = res_lmr.p_value
    p_values_w_chisq[i] = res_wc['p_value']

    np.savetxt('p_values_boostrap.txt',p_values_bootstrap,fmt='%0.8f',delimiter='\n')
    np.savetxt('p_values_lmr.txt',p_values_lmr,fmt='%0.8f',delimiter='\n')
    np.savetxt('p_values_w_chisq.txt',p_values_w_chisq,fmt='%0.8f',delimiter='\n')





