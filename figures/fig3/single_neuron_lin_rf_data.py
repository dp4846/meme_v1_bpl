# %% get linear receptive fields of all neurons for all natural image recordings
import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.io as sio
import os 

data_dir = '/scratch/gpfs/dp4846/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'

fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn]
sub_sample = 1

dim_filter_basis = 100 #number of image PCs to use in regression (regression gets noisier with more PCS, we did not find performance improvements beyond ~100)
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
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
    
    Y_m = r.mean('rep')
    Y_m = Y_m - Y_m.mean('stim')
    Y_m = Y_m.transpose('stim', 'unit').values
    beta_neur = np.linalg.lstsq(filter_resp, Y_m, rcond=None)[0]
    neur_filters = np.einsum('...ijk,...il->...jkl', filter_pc, beta_neur)
    np.save(eig_tuning_dir + 'neur_rf_est_' + fn + '.npy', neur_filters)