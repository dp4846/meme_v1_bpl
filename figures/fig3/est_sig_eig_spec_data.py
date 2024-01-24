#%%
import os
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import os
from itertools import product
from tqdm import tqdm
import datetime
import sys
sys.path.append('../../src/')
import eig_mom as em

data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
raw_data_dir =  data_dir + '/orig_stringer2019_data/'

#%%
fns = [fn for fn in os.listdir(resp_data_dir) if 'natimg2800_' in fn 
            and not 'npy' in fn and 'ms' in fn and '.nc' in fn and '0_M' in fn]
#%%
#first make data structures to hold results
n_neurs = []
sub_samp = 1
n_rec = len(fns)
# for holding onto eigenspectra raw results
fn_nms = [fns[rec].split('/')[-1].split('.')[0] for rec in range(n_rec)]
for rec in fn_nms:
    n_neurs.append(xr.open_dataset(resp_data_dir + rec + '.nc')['resp'][..., ::sub_samp].shape)
max_n_neurs = np.max(n_neurs)#the number of neurons in the largest recording
#this is the number of eigenvalues we will be able to estimate
#so xarray has to be this big
#eigenspectra will be estimated for each of the estimators below
pl_fit_types = ['cvpca', 'fit_cvpca_w', 'fit_cvpca_no_w',
                'pl_b0_raw', 'pl_b1_raw',]
#for holding onto parameters
nms = ['log_c1', 'alpha1', 'b1', 'alpha2', ]#names of parameters
pl_param_nms = []
for fit_type in pl_fit_types:#append fit type to parameter names
    if 'fit_cvpca' in fit_type:
        pl_param_nms += [fit_type + '_' + nm  for nm in ['log_c1', 'alpha1']]
    if 'pl' in fit_type:
        if 'b0' in fit_type:#if no breaks
            pl_param_nms += [fit_type + '_' + nm  for nm in nms[:2]]#only two params
        if 'b1' in fit_type:# if one break
            pl_param_nms += [fit_type + '_' + nm  for nm in nms[:4]]#four params

res_df = pd.DataFrame(np.zeros((n_rec, len(pl_param_nms))), columns=pl_param_nms)
#set index of res_df to be the recording names
res_df.index = fn_nms

# %% eigenmoment distribution estimation for weighting eigenmoments
#for holding onto eigenmoment mean and std
k_moms = 10
#to store estimated eigenmoments
mom_est = xr.DataArray(np.zeros((n_rec, k_moms)), 
                dims=['recording', 'mom'],
                coords=[fn_nms, range(1, 1 + k_moms),],)
#to store bootstrap samples of eigenmoment distributions
n_bs_samps = em.n_obs_needed_to_whiten_d_dims(k_moms)
mom_dist = xr.DataArray(np.zeros((n_rec, n_bs_samps, k_moms, )),
                dims=['recording', 'bs_samp', 'mom', ],
                coords=[fn_nms, range(n_bs_samps), range(1, 1 + k_moms),],)

sub_sample = 1#this is just for debugging, to run without the full data set
run_mom_est = False #this will ignore any mom_est and mom_dist files that already exist
#check if mom_dist and mom_est files already exist
if os.path.isfile(data_dir + './mom_dist.nc') and not run_mom_est:
    mom_dist = xr.open_dataarray('./mom_dist.nc')
    mom_est = xr.open_dataarray('./mom_est.nc')
else:
    for rec in tqdm(list(mom_dist.coords['recording'].values)):
        # print recording file name
        print(rec)
        #load data
        ds = xr.open_dataset(resp_data_dir + rec + '.nc')
        resp = ds['resp'][..., ::sub_sample]
        resp = resp/(resp[0]*resp[1]).mean('stim').sum('unit')**.5
        #estimate eigenmoments
        mom_est.loc[rec] = em.signal_er_eig_momk_centered(resp[0].values, resp[1].values, k_moms)
        #bootstrap eigenmoment distribution
        mom_dist.loc[rec] = em.bs_eig_mom(resp.values, k_moms, n_bs_samps)

    if sub_sample ==1:    
        mom_dist.to_netcdf('./mom_dist.nc' )
        mom_est.to_netcdf('./mom_est.nc' )
    else:#this was just for debugging
        mom_dist.to_netcdf('./mom_dist_sub_samp_' + str(sub_sample) + '.nc' )
        mom_est.to_netcdf('./mom_est_sub_samp_' + str(sub_sample) + '.nc' )

# %% eigenspectra parameter estimation using cvPCA and MEME
trans_type = 'raw'
do_cvPCA = True
do_meme = True
init_slope = 0.01
n_breaks = 50
k_moms_fit = 10
for rec in tqdm(list(res_df.index.values)):

    #cvPCA estimated power law
    ds = xr.open_dataset(resp_data_dir + rec + '.nc')
    Y_r = ds['resp'][..., ::sub_samp]
    Y_r = Y_r/(Y_r[0]*Y_r[1]).mean('stim').sum('unit')**.5 #rescale by estimate of total signal variance
    if do_cvPCA:

        Y_r = Y_r.values
    n_rep,  n_stim, n_neur = Y_r.shape

    if do_cvPCA:
        #get the cvpca raw estimates
        cvpca = em.cvpca(Y_r, force_psd=False)
        pl_fit_ind = np.arange(11, 500).astype(int)#indices aliong which to fit power law

        for weight in [True, False]:
            #fit power law to cvpca estimates
            ypred, b = em.get_powerlaw(cvpca, pl_fit_ind, weight=weight)
            alpha_1, log_c1 = b
            if weight:
                res_df.loc[rec, ['fit_cvpca_w_log_c1', 'fit_cvpca_w_alpha1',]] = log_c1, alpha_1

            else:
                res_df.loc[rec, ['fit_cvpca_no_w_log_c1', 'fit_cvpca_no_w_alpha1',]] = log_c1, alpha_1

    #print finished cvpca
    print('finished cvpca')
    if do_meme:
        init_break_points = list(range(2, 50, 2))
        search_break_points = init_break_points + list(np.logspace(np.log10(50), np.log10(n_neur),
                                        n_breaks-len(init_break_points)).astype(int))
        ######### MEME
        ind = np.arange(1, n_neur + 1)
        # load  estimated eigenmoments and distribution
        est_eig_mom = mom_est.loc[rec][:k_moms_fit].values
        bs_est_eig = mom_dist.loc[rec][..., :k_moms_fit].values
        #weighting matrix according to eigenmoment covariance for MEME
        W = em.bs_eig_mom_cov_W(transform=trans_type, bs_est_eig=bs_est_eig, return_W=True)

        ######## one slope power-law
        slopes = [init_slope, ]# initial guess at slope
        break_points = []# no breaks
        init_log_c1 = np.log(1/n_neur)
        res = em.fit_broken_power_law_meme_W(Y_r, k_moms_fit, break_points,
                                                init_log_c1, slopes,
                                                return_res=False, W=W, 
                                                transform=trans_type,
                                                est_eig_mom=est_eig_mom)
        log_c1, alpha_1 = res
        res_df.loc[rec, ['pl_b0_' + trans_type + '_log_c1',
                            'pl_b0_' + trans_type + '_alpha1',]] = log_c1, alpha_1
        print('finished pl_b0')

        ######### two slope power-law
        slopes = [init_slope, ]*2# initial guess at slope
        search_break_points_list = [[break_point,] for break_point in search_break_points]
        params, b1 = em.break_point_search_fit_broken_power_law_meme(Y_r, k_moms_fit, 
                                        search_break_points_list, init_log_c1, slopes,
                                        return_res=False, W=W, transform='raw',
                                        bs_est_eig=bs_est_eig, est_eig_mom=est_eig_mom)

        log_c1, alpha_1, alpha_2 = params

        break_points = [b1,]
        res_df.loc[rec, ['pl_b1_' + trans_type + '_log_c1',
                            'pl_b1_' + trans_type + '_alpha1',
                            'pl_b1_' + trans_type + '_b1',
                            'pl_b1_' + trans_type + '_alpha2',]] = log_c1, alpha_1, b1[0], alpha_2
        # print finished two slope power-law
        print('finished pl_b1')


    res_df.to_csv('str_pt_estimates.csv')



