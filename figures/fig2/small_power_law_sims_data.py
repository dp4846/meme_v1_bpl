#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 11:08:19 2022

@author: dean
"""
#%%
import sys
import numpy as np
from tqdm import tqdm
import os
from itertools import product
import pandas as pd
import xarray as xr
sys.path.append('../../src/')
import eig_mom as em

#%%
np.random.seed(2)
n_stim = 2000
d_neurons = 1000
n_rep = 2
c_sig = 1.
c_noise = 0.25
alpha_sig = 1.0
alpha_noise = 1.0
align=False
k_moms = 10
params = [[c_sig, c_noise, alpha_sig, alpha_noise, 500, align],
          [c_sig, 5.0, alpha_sig, alpha_noise, 500, align],]

param_df = pd.DataFrame(columns=['meme_logc', 'meme_alpha', 'c_sig', 'c_noise', 
                                'alpha_sig', 'alpha_noise', 'n_stim', 'd_neurons'],
                                index=range(len(params)))
eig_da = xr.DataArray(np.zeros((len(params), d_neurons, 4)),
                      coords=[range(len(params)), range(1, 1 + d_neurons), ['meme', 'cvpca', 'true_sig', 'true_noise']],
                      dims=['param', 'neuron', 'type'])

for i in tqdm(range(len(params))):
    c_sig, c_noise, alpha_sig, alpha_noise, n_stim, align = params[i]
    Y_r, ind, s_eig, n_eig = em.get_power_law_SN_sim(c_sig, c_noise, alpha_sig,
                                                  alpha_noise, d_neurons, n_stim,
                                                  n_rep, align, return_params=True)

    # need to estimate their eigenslopes and intercepts using stringer
    cvpca = em.cvpca(Y_r)

    meme_log_c1, meme_alpha1 = em.fit_broken_power_law_meme_W(Y_r, k_moms=k_moms, break_points = [], log_c1 = 0.0, slopes = [0.001,],)
    param_df.loc[i] = [ meme_log_c1, meme_alpha1, c_sig, c_noise,
                          alpha_sig, alpha_noise, n_stim, d_neurons]
    eig_da.loc[i, :, 'meme'] = np.exp(meme_log_c1)*ind**(-(meme_alpha1))
    eig_da.loc[i, :, 'cvpca'] = cvpca
    eig_da.loc[i, :, 'true_sig'] = s_eig
    eig_da.loc[i, :, 'true_noise'] = n_eig

# check if eig_da exists if so delete it and save new one
if os.path.exists('./small_power_law_sims.nc'):
    os.remove('./small_power_law_sims.nc')
if os.path.exists('./small_power_law_sims_params.csv'):
    os.remove('./small_power_law_sims_params.csv')

eig_da.to_netcdf('./small_power_law_sims.nc')
param_df.to_csv('./small_power_law_sims_params.csv')

# %%
