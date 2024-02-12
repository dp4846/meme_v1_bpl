# %%parametric bootstrap just for bpl ci's
import xarray as xr
import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.stats import multivariate_normal as mn
sys.path.append('../../src/')
import eig_mom as em
data_dir = '../../data/stringer_2019/'
raw_data_dir =  data_dir + '/processed_data/neural_responses/'
cov_data_dir = data_dir + '/processed_data/stringer_sn_covs/'
ci_dir = './bpl2_sims/'

n_bs = 20
init_slope = 0.01
init_log_c1 = 0.0
k_moms = 10
n_breaks = 50
initial_break_search_break_points = list(range(2, 50, 2))

# Load CSV with the first column as the index
fit_df = pd.read_csv('str_pt_estimates.csv')


#if you don't want to use threads (uses more memory but is faster) you can use the following
for rec in tqdm(range(7)):
    nm = fit_df['recording'][rec]
    fn = raw_data_dir + nm + '.nc'
    
    #check if you have already run this and start from last completed bootstrap
    write_nm = f"{ci_dir}rec_{rec}_{nm}_pbci_bpl2.csv"
    start_bs = 0
    if os.path.exists(write_nm):
        df = pd.read_csv(write_nm, index_col=0)
        last_valid = df.iloc[:,-1].last_valid_index()
        if last_valid is None:
            start_bs = 0
        else:
            start_bs = last_valid + 1
        print(f"loading {write_nm} and starting at {start_bs}")
    else:
        df = pd.DataFrame(columns=params.index, index=range(n_bs))
        df.to_csv(write_nm)

    ds = xr.open_dataset(fn)['resp']
    n_rep,  n_stim, n_neur = ds.shape
    da = xr.open_dataset(cov_data_dir + nm +'.nc')['est_mean_SNR_cov_stringer']
    sig_cov, sig_v, noise_cov = [da[i].values for i in range(3)]

    # load power law  to simulate from
    params = fit_df.set_index('fn_nms').loc[nm, ['pl_b1_raw_log_c1', 'pl_b1_raw_alpha1', 'pl_b1_raw_b1', 'pl_b1_raw_alpha2']]
    A = list(params[['pl_b1_raw_alpha1', 'pl_b1_raw_alpha2']])
    B = [int(params['pl_b1_raw_b1']),]
    log_c1 = params['pl_b1_raw_log_c1']
    ind = np.arange(1, 1 + n_neur).astype(int)
    log_cs = [log_c1,]
    sig_eig = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))


    #rescale appropriately
    resp = ds[..., :n_neur]
    # in the fitting of eigenmoments we rescaled by the sum of mean signal variances
    #see eig_s
    # construct signal covariance matrix
    sig_eig_sc = sig_eig 
    sig_cov_sc_pl = sig_v @ np.diag(sig_eig_sc) @ sig_v.T
    mean = resp.mean('stim').mean('rep').values

    #generate parameteric bootstrap data
    noise = mn.rvs(mean=[0,]*n_neur, cov=noise_cov, 
            size=(n_bs, 2, n_stim))
    sig = mn.rvs(mean=mean, cov=sig_cov_sc_pl, size=n_stim)      
    Y_r = sig[np.newaxis] + noise

    slopes = [init_slope, ]*2# initial guess at slope
    search_break_points = initial_break_search_break_points + [int(break_point) for break_point 
                                in np.logspace(np.log10(50), 
                                np.log10(n_neur), 
                                n_breaks-len(initial_break_search_break_points))]
    search_break_points_list = [[break_point,] for break_point in 
                                            search_break_points]
    #now fit each bootstrap
    for bs_ind in tqdm(range(start_bs, n_bs)):
        
        params, b1 = em.break_point_search_fit_broken_power_law_meme(Y_r[bs_ind], k_moms, 
                                        search_break_points_list, init_log_c1, slopes,
                                        return_res=False)

        log_c1, alpha_1, alpha_2 = params
        df.loc[bs_ind][['pl_b1_raw_log_c1', 'pl_b1_raw_alpha1', 'pl_b1_raw_b1', 'pl_b1_raw_alpha2']] = [log_c1, alpha_1, b1[0], alpha_2]
        df.to_csv(write_nm)