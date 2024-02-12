#%%
import xarray as xr
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from scipy.stats import multivariate_normal as mn
import sys
sys.path.append('../../src/')
import eig_mom as em

data_dir = '../../data/stringer_2019/'
raw_data_dir =  data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
cov_data_dir = data_dir + 'processed_data/stringer_sn_covs/'
fig3_dir = '../fig3/'#this is to get original estimates from data
sim_dir = 'meme_vs_cvpca_pl_est_match_str_sim/'
if not os.path.exists(sim_dir):
    os.makedirs(sim_dir)
    
np.random.seed(42)
n_bs = 20#how many simulations to run per recordings
init_slope = 0.01
k_moms = 10
fit_df = pd.read_csv(fig3_dir + 'str_pt_estimates.csv')

for rec in range(7):
    nm = fit_df['recording'][rec]
    fn = resp_data_dir + nm + '.nc'
    ds = xr.open_dataset(fn)['resp']
    n_rep,  n_stim, n_neur = ds.shape
    #get estimated signal and noise covariance matrices
    da = xr.open_dataset(cov_data_dir + nm +'.nc')['est_mean_SNR_cov_stringer']
    sig_cov, sig_v, noise_cov = [da[i].values for i in range(3)]
    #set power law parameters from direct fits estimated in fig 3
    ground_truth_params = fit_df.set_index('recording').loc[nm, ['fit_cvpca_w_log_c1', 'fit_cvpca_w_alpha1', ]]
    A = list(ground_truth_params[['fit_cvpca_w_alpha1', ]])
    log_c1 = ground_truth_params['fit_cvpca_w_log_c1']
    ind = np.arange(1, 1 + n_neur).astype(int)
    log_cs = [log_c1,]
    B = []
    sig_eig = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))
    resp = ds[..., :n_neur]
    # construct signal covariance matrix
    sig_eig_sc = sig_eig #this will be ground truth eigenspectra
    sig_mean =  resp.mean('stim').mean('rep').values


    #set up  dataframe that saves results
    est_sim_params = ['pl_b0_raw_log_c1',	'pl_b0_raw_alpha1', 'fit_cvpca_w_log_c1', 'fit_cvpca_w_alpha1', ]
    # now append these with three covariance conditions orig_SN_cov, aligned_SN_cov, and ind_SN_cov
    cov_params = ['orig', 'aligned', 'ind']
    est_sim_params = [i + '_SN_cov_' + j for i in est_sim_params for j in cov_params]

    # create a file for storing n_bs runs of the bootstrap with same columns as  params
    df = pd.DataFrame(columns=est_sim_params)

    #create initial csv file
    df['nm'] = nm
    write_nm = f"{sim_dir}rec_{rec}_{nm}_pbci_pl.csv"
    df.to_csv(write_nm)

    slopes = [init_slope, ]# initial guess at slope
    break_points = []# no breaks
    noise_eig = np.linalg.svd(noise_cov, compute_uv=False)#noise eig will be kept the same for simulations
    for cov_type in tqdm(['orig', 'aligned', 'ind']):
        if cov_type == 'orig':
            sig_cov_sim = sig_v @ np.diag(sig_eig_sc) @ sig_v.T # create signal covariance matrix with specified eigenvalues
            noise_cov_sim = noise_cov.copy()
            print('orig')
        elif cov_type == 'aligned':
            sig_cov_sim = sig_v @ np.diag(sig_eig_sc) @ sig_v.T
            noise_cov_sim = sig_v @ np.diag(noise_eig) @ sig_v.T
            noise_cov_sim = em.get_near_psd(noise_cov_sim)
            print('aligned')
        elif cov_type == 'ind':
            re_sample_ind = np.random.choice(n_neur, replace=False, size=n_neur)
            sig_cov_sim = sig_v[:, re_sample_ind] @ np.diag(sig_eig_sc) @ sig_v[:, re_sample_ind].T
            noise_cov_sim = noise_cov.copy()
            print('ind')

        noise = mn.rvs(mean=[0,]*n_neur, cov=noise_cov_sim, 
                size=(n_bs, 2, n_stim))
        sig = mn.rvs(mean=sig_mean, cov=sig_cov_sim, size=n_stim)     

        Y_r = sig[np.newaxis] + noise
        init_log_c1 = 0.
        for bs_ind in (range(n_bs)):
            #try meme
            #in original fitting used twice as many eigenmoments see est_sig_eig_spec
            n_samps = em.n_obs_needed_to_whiten_d_dims(int(k_moms * 2))
            W = em.bs_eig_mom_cov_W(Y_r=Y_r[bs_ind], k_moms=k_moms, n_samps=n_samps, 
                                    return_W=True,)
            res = em.fit_broken_power_law_meme_W(Y_r[bs_ind], k_moms, break_points,
                                                    init_log_c1, slopes,
                                                    return_res=True, W=W)
            log_c1, alpha_1 = res.x
            print('bs_ind '+ str(bs_ind))
            print(res)
            #save b0 results
            df.loc[bs_ind]['pl_b0_raw_log_c1' + '_SN_cov_' + cov_type , 
                        'pl_b0_raw_alpha1' + '_SN_cov_' + cov_type] = log_c1, alpha_1
            #try cvpca
            cvpca = em.cvpca(Y_r[bs_ind], force_psd=False)
            pl_fit_ind = np.arange(11, 500).astype(int)#indices along which to fit power law

            #fit power law to cvpca estimates
            ypred, b = em.get_powerlaw(cvpca, pl_fit_ind, weight=True)
            alpha_1, log_c1 = b
            
            df.loc[bs_ind].loc[['fit_cvpca_w_log_c1' + '_SN_cov_' + cov_type , 
                                'fit_cvpca_w_alpha1' + '_SN_cov_' + cov_type ,]] = log_c1, alpha_1
            df.to_csv(write_nm)

# %%
