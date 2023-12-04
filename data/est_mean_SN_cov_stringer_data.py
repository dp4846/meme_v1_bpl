#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 09:52:24 2022

@author: dean
"""
#%%
import os
import sys
sys.path.append('../src/')
import eig_mom as em
import xarray as xr
import numpy as np
from tqdm import tqdm
#go to the data directory and read in data_dir text file to get data_dir
with open('./data_dir.txt', 'r') as file:
    data_dir = file.read()
resp_data_dir = data_dir + 'processed_data/neural_responses/' 
cov_data_dir = data_dir + 'processed_data/stringer_sn_covs/'
fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if '.nc' in fn and 'ms' in fn]

#%%
sub_sample = 1
for fn in tqdm(fns):
    resp = xr.open_dataset(fn)['resp'][..., ::, ::sub_sample]
    resp = resp - resp.mean('stim')
    X = resp.values
    r_repeat, n_stim, n_neur = X.shape

    #make the covariance matrices for the two methods
    sig_cov = em.calc_sig_cov(resp, force_psd=True)
    #SVD will be used to set the eigenspectrum of the signal eigenspectrum
    w, sig_v = np.linalg.eigh(sig_cov)
    noise_cov = em.calc_noise_cov(resp)
    mats = np.concatenate([a[np.newaxis] for a in [sig_cov, sig_v, noise_cov]], 0)
    da = xr.DataArray(mats, dims=['mat_type', 'neuron_r', 'neuron_c'],
                 coords=[['sig_cov', 'sig_v',  'noise_cov',],
                         range(n_neur), range(n_neur)], 
                         name='est_mean_SNR_cov_stringer')
    #subsample is mainly used for testing 
    if sub_sample>1:
        save_file_nm = cov_data_dir + fn.split('/')[-1].split('.')[0] + '_sub_samp.nc'
        # if save file already exists, delete it and save new one
        if os.path.exists(save_file_nm):
            os.remove(save_file_nm)
        da.to_netcdf(save_file_nm)
    else:
        save_file_nm = cov_data_dir + fn.split('/')[-1].split('.')[0] + '.nc'
        # if save file already exists, delete it and save new one
        if os.path.exists(save_file_nm):
            os.remove(save_file_nm)
        da.to_netcdf(cov_data_dir + fn.split('/')[-1].split('.')[0] + '.nc')

