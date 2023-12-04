#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:08:19 2022

@author: dean
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import sys
sys.path.append('../../src/')
import eig_mom as em


#%% FIG 2 C,D - power law fits small simulations
eig_da = xr.open_dataarray('./small_power_law_sims.nc')
param_df = pd.read_csv('./small_power_law_sims_params.csv', index_col=0)
titles = ['Low noise',  'High noise']
s = 1
fig, ax = plt.subplots(1, len(param_df), figsize=(6*s,3*s), dpi=100)
for i, j in enumerate([0,1,]):
    meme, cvpca, true_sig, true_noise = eig_da.loc[j, :, :].values.T
    ind = eig_da.coords['neuron'].values
    ax[i].plot(ind, true_sig, c='grey', lw=2, label=r'Signal eigenspectra $\lambda_S$')
    ax[i].plot(ind, true_noise, c='black', lw=1, label = r'Noise eigenspectra $\lambda_\epsilon$',)
    ax[i].plot(ind, cvpca, c='blue', alpha=0.2, label = 'cvPCA')
    #pl_eigs, pl_params = em.get_powerlaw(cvpca, trange=ind-1, weight=False)
    #ax[i].plot(ind, pl_eigs, c='blue', ls='--', label='cvPCA power-law fit\non indices 1-1000')
    ax[i].loglog();
    ax[i].axis('square')
    ax[i].set_title(titles[j])
    pred_ind_end = 50
    pred_inds = np.arange(1, pred_ind_end).astype(int)
    pl_eigs, pl_params = em.get_powerlaw(cvpca, trange=pred_inds, weight=False)
    ax[i].plot(ind, pl_eigs, c='blue', ls=':', label='cvPCA power-law fit on\nindices 2-' + str(pred_ind_end))
    ax[i].plot(ind, meme, c='red', ls=':', label='MEME power-law fit')

    if i==1:
        ax[i].legend(framealpha=1, loc=(1.1,0))
    if i ==0:
        ax[i].set_xlabel('Eigenvalue index')
        ax[i].set_ylabel('Eigenvalue')

    else:
        ax[i].set_xticklabels([]);ax[i].set_yticklabels([]);

    ax[i].set_xlim(0.5,1500)
    ax[i].set_ylim(1e-4,1e1)
    ax[i].set_xticks([1, 10, 100, 1000])
    ax[i].grid()
fig.tight_layout()
plt.savefig('pl_example_sims.pdf', bbox_inches='tight', transparent=True)


# %%
