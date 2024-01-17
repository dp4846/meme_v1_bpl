
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:02:46 2021

@author: dean
"""
#%%
import numpy as np 
import eig_mom as em
from scipy.stats import ttest_1samp
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_mom(x, k_moms):
    m_p = np.array([np.mean(x**p) for p in range(1, 1 + k_moms)])
    return m_p
#%% test that eigenenmoment estimation is unbiased
n_sims = 50#number of simulations to run, more sims more sensitive to bias
k_moms = 10#number of eigenmoments to test estimates of

#simulation parameters
n_rep = 2#number of repetitions of stimulus
c_sig = 1.#signal strength
c_noise = 0.1#noise strength
alpha_sig = 1.#signal power law exponent
alpha_noise = 1.#noise power law exponent
d_neurons = 100#number of neurons
n_stim = 1000#number of stimuli
align = True#align signal and noise eigenvecs

#run simulations of data from distribution defined by parameters above
Y_rs = []
for i in range(n_sims):
    Y_r, ind, s_eig, n_eig = em.get_power_law_SN_sim(c_sig, c_noise, 
                                                    alpha_sig, alpha_noise, 
                                                    d_neurons, n_stim, n_rep, align,
                                                    return_params=True)
    Y_rs.append(Y_r)
Y_rs = np.array(Y_rs)

# now estimate signal eigenmoments and test that they are unbiased
res = []
for i in tqdm(range(n_sims)):
    est_sig_mom = em.signal_er_eig_momk_centered(Y_rs[i, 0], Y_rs[i, 1], k_moms)
    res.append(est_sig_mom)
res = np.array(res)
true_sig_mom = get_mom(s_eig, k_moms)#true signal eigenmoments
t, p = ttest_1samp(res, true_sig_mom[np.newaxis], axis=0)#t-test of eigenmoment estimates different from true eigenmoments
assert np.all(p > 0.001/len(p))
#plot eigenmoment estimates with error bars and true eigenmoments
plt.figure(figsize=(3, 3))
plt.errorbar(range(1, 1 + k_moms), np.mean(res, axis=0), 
yerr=1.96*np.std(res, axis=0)/n_sims**0.5, c='r', label='est+- SEM')
plt.plot(range(1, 1 + k_moms), true_sig_mom, 'k', label='true')
plt.legend()
plt.xlabel('eigenmoment index')
plt.ylabel('signal eigenmoment')
plt.semilogy()
# %% test that power-law fit to eigenmoments works
# have a break point but the true power-law has no break point two slopes should be similar
i = 0 #simulation index
break_points = [10,]#
log_c1 = 0.
slopes = [0.1, 2.,]# wrong slopes for init guess
Y_r = Y_rs[i]
res = em.fit_broken_power_law_meme_W(Y_r, k_moms, break_points, log_c1, slopes,
                                 return_res=True, W=None, transform='raw',
                                 bs_est_eig=None)
assert(res.status)
#plot true and estimated eigenvalues
est_s_eig = em.get_bpl_func_all(A=res.x[1:], log_c1=res.x[0], B=break_points, ind=ind)
plt.figure(figsize=(3, 3))
plt.plot(ind, s_eig, 'k', label='true')
plt.plot(ind, np.exp(est_s_eig), 'r', label='est', ls='--')
plt.legend()
plt.xlabel('eigenvalue index')
plt.ylabel('signal eigenvalue')
plt.loglog()
# %% test break point search estimator (just making sure break point search doesn't break)
slopes = [0.1, 2.,]
log_c1 = 0.1
break_points = [10,]
all_break_points = [[5,], [10,]] # choose between these two break points (arbitrary b/c no break)
bpl = em.get_bpl_func_all(A=slopes, log_c1=log_c1, B=break_points, ind=ind)
#returns break point and associated slopes with lowest error
res_exc = em.break_point_search_fit_broken_power_law_meme(Y_r, k_moms, all_break_points, log_c1, slopes,
                                 return_full_res=False, W=None)
#plot true and estimated eigenvalues
est_s_eig = em.get_bpl_func_all(A=res_exc[0][1:], log_c1=res_exc[0][0], B=res_exc[1], ind=ind)
plt.figure(figsize=(3, 3))
plt.plot(ind, s_eig, 'k', label='true')
plt.plot(ind, np.exp(est_s_eig), 'r', label='est', ls='--')
plt.legend()
plt.xlabel('eigenvalue index')
plt.ylabel('signal eigenvalue')
plt.loglog()

# %%
