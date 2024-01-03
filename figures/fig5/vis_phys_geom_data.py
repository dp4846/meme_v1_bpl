import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.io as sio
import os 
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
def noise_corr_R2(X, Y, noise_corrected=True):
    #X is observation x feature
    #Y is repeat x observation X neuron
    K, M, N = Y.shape
    M, D = X.shape
    Y_m = Y.mean(0)
    Y_m = Y_m - Y_m.mean(0, keepdims=True)#subtract mean response (could include intercept in model)
    hat_beta = np.linalg.lstsq(X, Y_m, rcond=None)[0]#regression of PCs of images on single units
    hat_r = X @ hat_beta#predicted responses from linear transform of images
    rss = ((Y_m - hat_r)**2).sum(0)/(M-D)#residual sum of squares
    var_r = Y_m.var(0, ddof=1) #estimate total variance of responses
    linear_var = var_r - rss #estimate of linear variance by subtracting residual variance from total variance

    if noise_corrected:
        S_var = (var_r - Y.var(0, ddof=1,).mean(0)/K)#signal variance
        return linear_var / S_var
    else:
        return linear_var / var_r
def hat_snr(x, noise_corrected=True):
    #x is a 2d array of shape (n_rep, n_stim, ...)
    n_rep, n_stim = x.shape[:2]
    noise_var = np.mean(np.var(x, 0, ddof=1), 0)
    sig_var = np.var(np.mean(x, 0), 0, ddof=0)
    snr = sig_var/noise_var#raw SNR estimate
    snr_corr = (sig_var - ((n_stim-1)/n_stim)*noise_var/n_rep)/noise_var #SNR estimate corrected for finite number of trials
    if noise_corrected:
        return snr_corr
    else:
        return snr

def create_2d_gabor(img_size, sigma, f, theta, phi, center_x, center_y):
    """
    Create a 2D Gabor filter.

    Parameters:
    - img_size: Size of the output Gabor filter (square).
    - sigma: Standard deviation of the Gaussian envelope.
    - gamma: Aspect ratio that controls the ellipticity of the Gaussian envelope.
    - f: Spatial frequency of the cosine factor.
    - theta: Orientation of the Gabor filter in radians.
    - phi: Phase offset.

    Returns:
    - gabor_filter: 2D Gabor filter of the specified parameters.
    """

    x = np.linspace(-img_size // 2, img_size // 2, img_size) - center_x
    y = np.linspace(-img_size // 2, img_size // 2, img_size) - center_y
    x, y = np.meshgrid(x, y) 

    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gabor_real = np.exp(-0.5 * (x_theta**2 + y_theta**2) / (sigma**2)) * np.cos(2 * np.pi * f * x_theta + phi)
    return gabor_real

# function for making gabors
def make_gabor_filters(img_size=60, scales=[1, 0.5, 0.25, ], thetas=[0., 0.78, 1.57, 2.35]):
    inds = []
    filters = []
    for scale in tqdm(scales):
        sigma = scale * img_size / 4
        f = 1 / (scale * img_size / 2)
        steps_x = steps_y =  np.linspace(scale*img_size/2, img_size - scale*img_size/2, int(1/scale)) - img_size/2
        #increase number of steps so that there is half overlap
        if scale<1:
            steps_x = steps_y =  np.linspace(scale*img_size/2, img_size - scale*img_size/2, int(1./scale)) - img_size/2
        for step_x in steps_x:
            for step_y in steps_y:
                for theta in (thetas):
                    for phase in [0, np.pi/2]:
                        gabor_filter = create_2d_gabor(img_size, sigma, f, theta, phase, center_x=step_x, center_y=step_y)
                        filters.append(gabor_filter)
                        inds.append(np.array([scale, step_x, step_y, theta, phase,]))
    filters_gabor = np.array(filters)
    gab_inds = np.array(inds)
    print('number of gabors: ' + str(len(filters_gabor)))
    return filters_gabor, gab_inds

rf_coords = {'ms_natimg2800_M160825_MP027_2016-12-14':[(25, 65), (35, 85)],
             'ms_natimg2800_M161025_MP030_2017-05-29':[(25, 65), (95, 145)],
            'ms_natimg2800_M170604_MP031_2017-06-28':[(25, 65), (90, 135)],
            'ms_natimg2800_M170714_MP032_2017-08-07':[(20, 55), (20, 60)],
            'ms_natimg2800_M170714_MP032_2017-09-14':[(10, 45), (90, 130)],
            'ms_natimg2800_M170717_MP033_2017-08-20':[(5, 60), (90, 150)],
            'ms_natimg2800_M170717_MP034_2017-09-11':[(20, 60), (90, 135)]}

data_dir = '/scratch/gpfs/dp4846/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'

fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn]
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
sub_sample = 1

# %% SNR estimation for eigs and neurons
for rec in tqdm(range(len(fns))):
    da = xr.open_dataset(fns[rec])
    fn = fns[rec].split('/')[-1].split('.')[0]

    #SVD of signal covariance matrix for neurons and stimuli
    r = da['resp'][:, ::sub_sample, ::sub_sample]
    n_rep, n_stim, n_neur = r.shape
    r = r - r.mean('stim')
    r_train = r[:, :2000]#subset of stimuli on which to estimate signal eigenvectors
    r_test = r[:, 2000:]#subset of stimuli on which to estimate SNR
    n_rep_train, n_stim_train, n_neur_train = r_train.shape
    n_rep_test, n_stim_test, n_neur_test = r_test.shape
    sig_cov_train = r_train[0].values.T @ r_train[1].values # just using for SVD so doesn't need scaling to covariance estimate
    u_r_train = TruncatedSVD(n_components=n_stim_train, algorithm='arpack').fit(sig_cov_train).components_.T
    n_pc = n_stim_train
    snrs=[]
    for i in tqdm(range(n_pc)):
        #these are the two 'repeat' responses from the eigenmode
        u1 = u_r_train[:,i:i+1].T @ r_test[0].values.T#project test response from trial 1 onto trained singular vector
        u2 = u_r_train[:,i:i+1].T @ r_test[1].values.T#project test response from trial 2 onto trained singular vector
        u = np.concatenate([u1, u2], 0)
        snr_corr = hat_snr(u, noise_corrected=True)#calculate snr across repeats
        snrs.append(snr_corr)
    snrs_pc_train_split = np.array(snrs)
    snr_neurs = hat_snr(r, noise_corrected=True)#get snr for each neuron

    np.save(eig_tuning_dir + 'neur_snr_' + fn + '.npy', snr_neurs)
    np.save(eig_tuning_dir + 'pc_snr_' + fn + '.npy', snrs_pc_train_split)

# %% eig and neur r2 with linear filters effect of dimensionality and stimuli truncation
n_eigenmodes = 10
dims = [2, 4, 8, 32, 64, 128, 256, 512]
dim_filter_basis = np.max(dims)
data = np.zeros((len(fns), len(dims), 2, 2, n_eigenmodes))
fn_labels = [fn.split('/')[-1].split('.')[0] for fn in fns]

eig_pc_r2 = xr.DataArray(data, dims=['rec', 'dim', 'var_stab', 'rf_type', 'pc'],
                    coords={'rec':fn_labels,
                            'dim':dims,
                            'var_stab':[True, False],
                            'rf_type':['whole', 'truncated'],
                            'pc':np.arange(n_eigenmodes)})
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
for rec in tqdm(range(len(fns))):
    fn = fns[rec]
    r = xr.open_dataset(fn)['resp']
    n_rep, n_stim, n_neur = r.shape
    # create neuron dataarray
    neur_pc_r2 = xr.DataArray(np.zeros((len(dims), 2, 2, n_neur)), 
                              dims=['dim', 'var_stab', 'rf_type', 'unit'], 
                              coords={'dim':dims,
                                    'var_stab':[True, False],
                                    'rf_type':['whole', 'truncated'],
                                    'unit':np.arange(n_neur)})
    for rf_type in ['whole', 'truncated']:
        if rf_type=='whole':
            imgs_exp = imgs[:, :, r.coords['stim'].values]
        else:
            (r1, r2), (c1, c2) = rf_coords[fn.split('/')[-1].split('.')[0]]
            imgs_exp = imgs[r1:r2, c1:c2, r.coords['stim'].values]
        imgs_exp = imgs_exp.transpose((-1,0,1))#reshape for SVD
        n_imgs, n_rows, n_cols = imgs_exp.shape
        S = imgs_exp.reshape((n_imgs, n_rows*n_cols))#this gives you number images X number pixels
        # we get responses from 'PC' linear filters to images (with clever PCA tricks could have taken a shorter route).
        u_im, s_im, v_im = np.linalg.svd(S, full_matrices=True)#PCA of images
        filter_pc = v_im[:dim_filter_basis].reshape((dim_filter_basis,) + (n_rows, n_cols))
        filter_resp = np.einsum('ijk,ljk->il', imgs_exp, filter_pc)
        for var_stab in [True, False]:
            u_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn_labels[rec] + '.npy')[:, :n_eigenmodes]
            Y = r.transpose('rep', 'stim', 'unit').values
            if var_stab:
                u_r_stim = (u_r_stim - u_r_stim.min(0, keepdims=True))**0.5
                Y = (Y - Y.min(0, keepdims=True))**0.5
            for dim in dims:
                r2 = noise_corr_R2(filter_resp[:, :dim], u_r_stim[None], noise_corrected=False)
                eig_pc_r2.loc[{'rec':fn.split('/')[-1].split('.')[0], 'dim':dim, 'var_stab':var_stab, 'rf_type':rf_type}] = r2
                r2 = noise_corr_R2(filter_resp[:, :dim], Y, noise_corrected=True)
                neur_pc_r2.loc[{'dim':dim, 'var_stab':var_stab, 'rf_type':rf_type}] = r2
    neur_pc_r2.to_netcdf(eig_tuning_dir + 'neur_pc_r2_' + fn_labels[rec] + '.nc')
eig_pc_r2.to_netcdf(eig_tuning_dir + 'eig_pc_r2.nc')

#%%  find performance of gabor and their energy for each eigenmode
fn_labels = [fn.split('/')[-1].split('.')[0] for fn in fns]
n_eigenmodes = 10
eig_gabor_r2 = xr.DataArray(np.zeros((len(fns), 2, n_eigenmodes)), dims=['rec', 'var_stab', 'pc'],
                    coords={'rec':fn_labels,
                            'var_stab':[True, False],
                            'pc':np.arange(n_eigenmodes)})
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
for rec in tqdm(range(len(fns))): 
    fn = fns[rec].split('/')[-1].split('.')[0]
    (r1, r2), (c1, c2) = rf_coords[fn]
    max_dist = np.max([r2-r1, c2-c1])
    filters_gabor, gab_inds = make_gabor_filters(img_size=max_dist)
    r = xr.open_dataset(fns[rec])['resp']

    imgs_exp = imgs[r2-max_dist:r2, c2-max_dist:c2, r.coords['stim'].values]#get the images in order to correspond to the responses
    imgs_exp = imgs_exp.transpose((-1,0,1))
    filters_gabor = np.concatenate((np.ones((1, max_dist, max_dist)), filters_gabor), 0)
    filter_resp = np.einsum('ijk,ljk->il', imgs_exp, filters_gabor)
    filter_resp_sq = filter_resp**2
    filters_gabor_resp = np.concatenate((filter_resp, filter_resp_sq), 1)

    n_rep, n_stim, n_neur = r.shape

    neur_gabor_r2 = xr.DataArray(np.zeros((2, n_neur)), dims=['var_stab', 'unit'],
                    coords={'var_stab':[True, False],
                            'unit':np.arange(n_neur)})

    for var_stab in eig_gabor_r2.coords['var_stab']:
        var_stab = var_stab.values.item()
        u_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy')[:, :n_eigenmodes]
        Y = r.transpose('rep', 'stim', 'unit').values
        if var_stab:
            u_r_stim = (u_r_stim - u_r_stim.min(0, keepdims=True))**0.5
            u_r_stim = u_r_stim - u_r_stim.mean(0, keepdims=True)
            Y = (Y - Y.min(0, keepdims=True))**0.5
        r2 = noise_corr_R2(filters_gabor_resp, u_r_stim[None], noise_corrected=False)
        eig_gabor_r2.loc[{'rec':fn, 'var_stab':var_stab}] = r2
        r2 = noise_corr_R2(filters_gabor_resp, Y, noise_corrected=True)
        neur_gabor_r2.loc[{'var_stab':var_stab}] = r2
        
    neur_gabor_r2.to_netcdf(eig_tuning_dir + 'neur_gabor_r2_' + fn + '.nc')
eig_gabor_r2.to_netcdf(eig_tuning_dir + 'eig_gabor_r2.nc')
# %% plots of 32 gabor filters
# Select the first 32 images
filters_gabor, gab_inds = make_gabor_filters()
selected_images = filters_gabor[::3, :]
vmin = selected_images.min()
vmax = selected_images.max()
# Create a subplot grid
num_rows = 4
num_cols = 8
s= 0.5
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12*s, 6*s))
# Plot each image in the grid
for i in range(num_rows):
    for j in range(num_cols):
        index = i * num_cols + j
        if index < selected_images.shape[0]:
            img = selected_images[index]
            axes[i, j].imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
            axes[i, j].axis('off')
plt.suptitle('32 Gabor filters of ' + str(filters_gabor.shape[0]))
plt.tight_layout()
plt.savefig('./example_gabor_filters.pdf', dpi=300)