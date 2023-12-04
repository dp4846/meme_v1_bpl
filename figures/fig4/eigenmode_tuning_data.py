#%% 
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.io as sio
from sklearn.decomposition import TruncatedSVD
def hat_snr(x):
    #x is a 2d array of shape (n_rep, n_stim, ...)
    n_rep, n_stim = x.shape[:2]
    noise_var = np.mean(np.var(x, 0, ddof=1), 0)
    sig_var = np.var(np.mean(x, 0), 0, ddof=0)
    snr = sig_var/noise_var#raw SNR estimate
    snr_corr = (sig_var - ((n_stim-1)/n_stim)*noise_var/n_rep)/noise_var #SNR estimate corrected for finite number of trials
    return snr, snr_corr

data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'

fns = [resp_data_dir + fn for fn in os.listdir(orig_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
sub_sample = 10


#%%
#SNR estimation for PCs and neurons
for rec in tqdm(range(len(fns))):
    da = xr.open_dataset(fns[rec])
    fn = fns[rec].split('/')[-1].split('.')[0]

    #SVD of signal covariance matrix for neurons and stimuli
    r = da['resp'][:, ::sub_sample, ::sub_sample]/1e4   
    n_rep, n_stim, n_neur = r.shape
    r = r - r.mean('stim')
    r_train = r[:, ::2]
    r_test = r[:, 1::2]
    n_rep_train, n_stim_train, n_neur_train = r_train.shape
    n_rep_test, n_stim_test, n_neur_test = r_test.shape
    sig_cov_train = r_train[0].values.T @ r_train[1].values # (just using for SVD so doesn't need scaling)
    # u_r_train = np.linalg.svd(sig_cov_train, full_matrices=False)[0] #SVD of signal covariance matrix for neurons
    u_r_train = TruncatedSVD(n_components=n_stim_train, algorithm='arpack').fit(sig_cov_train).components_.T
    n_pc = n_stim_train
    snrs=[]#go through each singular vector and compute SNR from test responses by projecting onto that mode
    for i in tqdm(range(n_pc)):
        u1 = u_r_train[:,i:i+1].T @ r_test[0].values.T#project test response from trial 1 onto trained singular vector
        u2 = u_r_train[:,i:i+1].T @ r_test[1].values.T#project test response from trial 2 onto trained singular vector
        u = np.concatenate([u1, u2], 0)#calculate snr across trials
        snr, snr_corr = hat_snr(u)
        snrs.append([snr, snr_corr])
    snrs_pc_train_split = np.array(snrs)
    snr_neur, snr_neur_corr = hat_snr(r)
    snr_neurs = np.array([snr_neur, snr_neur_corr])

    np.save(eig_tuning_dir + 'neur_snr_' + fn + '.npy', snr_neurs)
    np.save(eig_tuning_dir + 'pc_snr_' + fn + '.npy', snrs_pc_train_split)

#%%
# svd of split cov matrix for neur X neur and stim X stim
for rec in tqdm(range(len(fns))):
    da = xr.open_dataset(fns[rec])
    fn = fns[rec].split('/')[-1].split('.')[0]
    r = da['resp'][:, ::sub_sample, ::sub_sample]/1e4   
    n_rep, n_stim, n_neur = r.shape
    r = r - r.mean('stim')
    sig_cov_neur = r[0].values.T @ r[1].values
    sig_cov_stim = r[0].values @ r[1].values.T
    u_r_neur = np.linalg.svd(sig_cov_neur, full_matrices=False)[0] 
    u_r_stim = np.linalg.svd(sig_cov_stim, full_matrices=False)[0]
    np.save(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy', u_r_stim)
    np.save(eig_tuning_dir + 'sp_cov_neur_u_r_' + fn + '.npy', u_r_neur)

#%%
#R^2 for each neuron and PC
for rec in tqdm(range(len(fns))):
    da = xr.open_dataset(fns[rec])
    fn = fns[rec].split('/')[-1].split('.')[0]
    #now load the saved u_r_stim to get the linearity of the eigenmodes
    u_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy')

    r = da['resp'][:, ::sub_sample, ::sub_sample]/1e4   
    n_rep, n_stim, n_neur = r.shape
    imgs = sio.loadmat(data_dir + 'images_natimg2800_all.mat')['imgs']
    imgs = imgs[:, :, r.coords['stim'].values]
    _ = imgs.transpose((-1,0,1))
    X = _.reshape((_.shape[0],np.product(_.shape[1:])))#this gives you number images X number pixels
    u_im, s_im, v_im = np.linalg.svd(X, full_matrices=False)#PCA of images
    dim = 10
    beta = np.linalg.lstsq(u_im[:, :dim], u_r_stim, rcond=None)[0]
    beta = np.array(beta)
    vs = ((np.diag(s_im) @ v_im))[:dim]
    rf = [(beta[:, i:i+1,] * vs).sum(0).reshape(imgs.shape[:2]) for i in range(beta.shape[0])]
    rf = np.array(rf)
    np.save(eig_tuning_dir + 'rf_est_' + fn + '.npy', rf)

    imgs = imgs.transpose((2,0,1))
    l_r = np.array([(rf[i:i+1]*imgs).sum((1,2)) for i in range(rf.shape[0])])
    betas = np.array([np.linalg.lstsq(l_r[i][np.newaxis].T, u_r_stim[:,i], rcond=None)[0] for i in range(l_r.shape[0])])

    np.save(eig_tuning_dir + 'linear_resp_betas_' + fn + '.npy', betas)
    np.save(eig_tuning_dir + 'lin_rf_resp_' + fn + '.npy', l_r)


    u_r_s = xr.DataArray(u_r_stim, dims=['stim', 'pc'], coords={'stim':r.coords['stim'].values,
                                                            'pc':np.arange(u_r_stim.shape[1])})
    r_mean = r.mean('rep')
    D = dim#number of PCs to use in regression
    M = X.shape[0]#number of images
    
    #regress images onto single units
    hat_beta = np.linalg.lstsq(u_im[:, :D], r_mean, rcond=None)[0]#regression of PCs of images on single units
    hat_r = u_im[:, :D] @ hat_beta#predicted responses from linear transform of images
    rss = ((r_mean - hat_r)**2).sum('stim')/(M-D)#residual sum of squares
    var_r = r_mean.var('stim', ddof=1)#estimate total variance of responses
    linear_var = var_r - rss#estimate of linear variance by subtracting residual variance from total variance
    L = linear_var.sum('unit')
    S_var = (var_r - r.var('rep', ddof=1, ).mean('stim')/2).sum('unit')#signal variance

    #regress images onto PCs across single units
    hat_beta = np.linalg.lstsq(u_im[:, :D], u_r_s[:, :], rcond=None)[0]
    hat_u_R = u_im[:,:D] @ hat_beta
    rss = ((u_r_s[:, :] - hat_u_R)**2.).sum('stim')/(M-D)
    var_u_r = u_r_s[:, :].var('stim', ddof=1)
    linear_var_K_pcs = var_u_r - rss
    r2s_pc = [linear_var_K_pcs[ind]/var_u_r[ind] for ind in range(len(linear_var_K_pcs))]
    r2s_neur = [linear_var[ind]/var_r[ind] for ind in range(len(var_r))]

    np.save(eig_tuning_dir + 'lin_r2_pc_' + fn + '.npy', r2s_pc)
    np.save(eig_tuning_dir + 'lin_r2_neur' + fn + '.npy', r2s_neur)

