#%%
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_beta(X, Y):
    #y is repeat x neuron x observation
    #x is observation x feature
    Y_m = (Y.mean(0) - Y.mean((0,1))).squeeze()
    hat_beta = np.linalg.lstsq(X, Y_m, rcond=None)[0]#regression of PCs of images on single units
    return hat_beta
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

fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
rf_pos_labs = {'ms_natimg2800_M160825_MP027_2016-12-14':[30, 90],
 'ms_natimg2800_M161025_MP030_2017-05-29':[90, 150],
 'ms_natimg2800_M170604_MP031_2017-06-28':[90, 150],
 'ms_natimg2800_M170714_MP032_2017-08-07':[30, 90],
 'ms_natimg2800_M170714_MP032_2017-09-14':[90, 150],
 'ms_natimg2800_M170717_MP033_2017-08-20':[90, 150],
 'ms_natimg2800_M170717_MP034_2017-09-11':[90, 150]}#figured out where to cut off RF by eye

rf_pos_labs = {'ms_natimg2800_M160825_MP027_2016-12-14':[30, 90],
 'ms_natimg2800_M161025_MP030_2017-05-29':[90, 150],
 'ms_natimg2800_M170604_MP031_2017-06-28':[90, 150],
 'ms_natimg2800_M170714_MP032_2017-08-07':[30, 90],
 'ms_natimg2800_M170714_MP032_2017-09-14':[90, 150],
 'ms_natimg2800_M170717_MP033_2017-08-20':[90, 150],
 'ms_natimg2800_M170717_MP034_2017-09-11':[90, 150]}#figured out where to cut off RF by eye


sub_sample = 1
eig_filters = []
for rec in tqdm(range(len(fns))):
    da = xr.open_dataset(fns[rec])
    fn = fns[rec].split('/')[-1].split('.')[0]
    #now load the saved u_r_stim to get the linearity of the eigenmodes
    u_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy')
    n_pcs = 32
    u_r_stim = u_r_stim/u_r_stim.std(0, ddof=1)

    r = da['resp'][:, ::sub_sample, ::sub_sample]  
    #drop neurons with no variance
    r = r[..., r.mean('rep').var('stim')>0]
    n_rep, n_stim, n_neur = r.shape
    r = (r - r.min(('rep','stim')))**0.5
    r = r - r.mean(('rep','stim'))
    r = r/r.mean('rep').std('stim')
    snr = hat_snr(r.values)[1]
    imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
    left, right = rf_pos_labs[fn]
    #imgs = imgs[:, left:right, r.coords['stim'].values]
    imgs = imgs[:, :, r.coords['stim'].values]
    _ = imgs.transpose((-1,0,1))
    S = _.reshape((_.shape[0], np.product(_.shape[1:])))#this gives you number images X number pixels
    #PCA basis
    u_im, s_im, v_im = np.linalg.svd(S, full_matrices=True)#PCA of images
    dim = 100
    filter_pc = v_im[:dim,:].reshape((dim,) + imgs.shape[:-1])

    # print shapes of filter_pc and  imgs
    print('filter_pc shape: ', filter_pc.shape)
    print('imgs shape: ', imgs.shape)


    #get the dot prodduct of filters with all images
    n_eig_modes = 125
    #filter_resp = np.sum(filter_pc[...,None]*imgs[None :60], (1,2))

    filter_resp= np.einsum('ijk,jkl->il', filter_pc[:dim], imgs)

    beta_eig = get_beta(filter_resp.T, u_r_stim[:, :n_eig_modes][None])
    eig_filter = (filter_pc[:, ..., None]*beta_eig.squeeze()[:, None,None, :]).sum(0)
    #now show all the filters in a grid
    plt.figure(figsize=(10,10))
    plt.title(fn)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(eig_filter[..., i], cmap='gray')
        plt.axis('off')
    eig_filters.append(eig_filter)
    plt.figure(figsize=(3,3))
    rf = (eig_filter**2).sum(-1)
    plt.imshow(rf, cmap='gray')
# %%
for eig_filter in eig_filters:
    plt.figure(figsize=(3,3))
    rf = (eig_filter[...,:100]**2).sum(-1)**0.5
    plt.imshow(rf, cmap='gray')
    #put more ticks on the x axis
# %%
for i, eig_filter in enumerate(eig_filters):
    plt.figure(figsize=(3,3))
    rf = (eig_filter[...,:10]**2).sum(-1)**0.5
    #get the indices of a square in each RF that contain the most energy
    x_prof = (rf.sum(1))/rf.sum()
    y_prof = (rf.sum(0))/rf.sum()

    mean_y_prof = y_prof*np.arange(y_prof.shape[0])
    mean_x_prof = x_prof*np.arange(x_prof.shape[0])
    com = (mean_y_prof.sum(), mean_x_prof.sum())

    std_y_prof = np.sqrt((y_prof*(mean_y_prof-np.arange(y_prof.shape[0]))**2).sum())
    std_x_prof = np.sqrt((x_prof*(mean_x_prof-np.arange(x_prof.shape[0]))**2).sum())
   
    y_l = int(com[0] - 2*std_y_prof)
    y_r = int(com[0] + 2*std_y_prof)
    x_l = int(com[1] - 2*std_x_prof)
    x_r = int(com[1] + 2*std_x_prof)
    #now use the max distance from left to right and top to bottom to get the square
    max_dist = max(y_r - y_l, x_r - x_l)
    y_r = y_l + max_dist
    x_r = x_l + max_dist
    #now plot the square
    plt.figure(figsize=(3,3))
    plt.imshow(rf, cmap='gray')
    plt.plot([x_l, x_l], [y_l, y_r], color='r')
    plt.plot([x_r, x_r], [y_l, y_r], color='r')
    plt.plot([x_l, x_r], [y_l, y_l], color='r')
    plt.plot([x_l, x_r], [y_r, y_r], color='r')
    #now plot the square from rf_pos_labs
    fn = fns[i].split('/')[-1].split('.')[0]
    left, right = rf_pos_labs[fn]
    #these are just the positions from left to right, vertical is always from 0 to 60
    plt.plot([left, left], [0, 60], color='b')
    plt.plot([right, right], [0, 60], color='b')

#%% now plot the eig_filters
    
for eig_filter in eig_filters:
    plt.figure(figsize=(10,5))
    plt.title(fn)
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.imshow(eig_filter[..., i], cmap='gray')
        plt.axis('off')
# %%

# %%

# %%

#%%
    