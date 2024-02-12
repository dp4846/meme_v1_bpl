#%% 
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
sys.path.append('../../src/')
import eig_mom as em


    
rf_coords = {'ms_natimg2800_M160825_MP027_2016-12-14':[(25, 65), (35, 85)],
             'ms_natimg2800_M161025_MP030_2017-05-29':[(25, 65), (95, 145)],
            'ms_natimg2800_M170604_MP031_2017-06-28':[(25, 65), (90, 135)],
            'ms_natimg2800_M170714_MP032_2017-08-07':[(20, 55), (20, 60)],
            'ms_natimg2800_M170714_MP032_2017-09-14':[(10, 45), (90, 130)],
            'ms_natimg2800_M170717_MP033_2017-08-20':[(5, 60), (90, 150)],
            'ms_natimg2800_M170717_MP034_2017-09-11':[(20, 60), (90, 135)]}

data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'

fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
sub_sample = 1

#
#%% SVD of split cov matrix for neur X neur and stim X stim
for rec in tqdm(range(len(fns))):
    da = xr.open_dataset(fns[rec])
    fn = fns[rec].split('/')[-1].split('.')[0]
    r = da['resp'][:, ::sub_sample, ::sub_sample]   
    n_rep, n_stim, n_neur = r.shape
    r = r - r.mean('stim')
    sig_cov_neur = r[0].values.T @ r[1].values
    sig_cov_stim = r[0].values @ r[1].values.T
    u_r_neur = np.linalg.svd(sig_cov_neur, full_matrices=False)[0] 
    u_r_stim = np.linalg.svd(sig_cov_stim, full_matrices=False)[0]
    np.save(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy', u_r_stim)
    np.save(eig_tuning_dir + 'sp_cov_neur_u_r_' + fn + '.npy', u_r_neur)

#%% eigenmode linear prediction and filters
fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn]
sub_sample = 1
dim_filter_basis = 100 #number of image PCs to use in regression (regression gets noisier with more PCS, we did not find performance improvements beyond ~100)
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
dim = 100
for rec in tqdm(range(len(fns))):
    fn = fns[rec].split('/')[-1].split('.')[0]
    r = xr.open_dataset(fns[rec])['resp']
    imgs_exp = imgs[:, :, r.coords['stim'].values]#get the images in order to correspond to the responses
    imgs_exp = imgs_exp.transpose((-1,0,1))#reshape for SVD
    n_imgs, n_rows, n_cols = imgs_exp.shape
    S = imgs_exp.reshape((n_imgs, n_rows*n_cols))#this gives you number images X number pixels
    # we get responses from 'PC' linear filters to images (with clever PCA tricks could have taken a shorter route).
    u_im, s_im, v_im = np.linalg.svd(S, full_matrices=True)#PCA of images
    filter_pc = v_im[:dim_filter_basis].reshape((dim_filter_basis,) + (n_rows, n_cols))
    # get response of pc filters to every img.
    filter_resp = np.einsum('ijk,ljk->il', imgs_exp, filter_pc)
    #load eigenmode tuning
    u_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy')
    #fit linear model
    beta = np.linalg.lstsq(filter_resp, u_r_stim, rcond=None)[0]
    #get the linear prediction
    linear_pred = filter_resp @ beta
    r2 = em.noise_corr_R2(filter_resp, u_r_stim[None], noise_corrected=False)
    eig_filters = np.einsum('...ijk,...il->...jkl', filter_pc, beta)
    np.save(eig_tuning_dir + 'eig_rf_est_' + fn + '.npy', eig_filters)
    np.save(eig_tuning_dir + 'eig_lin_resp_' + fn + '.npy', linear_pred)
    np.save(eig_tuning_dir + 'eig_pc_r2_est_' + fn + '.npy', r2)


