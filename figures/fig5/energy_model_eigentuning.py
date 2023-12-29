#%%
import os
import numpy as np
import xarray as xr
from tqdm import tqdm
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm
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
def hat_snr(x):
    #x is a 2d array of shape (n_rep, n_stim, ...)
    n_rep, n_stim = x.shape[:2]
    noise_var = np.mean(np.var(x, 0, ddof=1), 0)
    sig_var = np.var(np.mean(x, 0), 0, ddof=0)
    snr = sig_var/noise_var#raw SNR estimate
    snr_corr = (sig_var - ((n_stim-1)/n_stim)*noise_var/n_rep)/noise_var #SNR estimate corrected for finite number of trials
    return snr, snr_corr

def noise_corr_R2(X, Y,):
    #y is repeat x neuron x observation
    #x is observation x feature
    K, M, N = Y.shape
    M, D = X.shape

    Y_m = (Y.mean(0) - Y.mean((0,1))).squeeze()
    hat_beta = np.linalg.lstsq(X, Y_m, rcond=None)[0]#regression of PCs of images on single units
    hat_r = X @ hat_beta#predicted responses from linear transform of images
    rss = ((Y_m - hat_r)**2).sum(0)/(M-D)#residual sum of squares
    var_r = Y_m.var(0, ddof=1) #estimate total variance of responses
    linear_var = var_r - rss #estimate of linear variance by subtracting residual variance from total variance
    S_var = (var_r - Y.var(0, ddof=1,).mean(0)/K)#signal variance
    return linear_var / S_var, linear_var / var_r

def get_beta(X, Y):
    #y is repeat x neuron x observation
    #x is observation x feature
    Y_m = (Y.mean(0) - Y.mean((0,1))).squeeze()
    hat_beta = np.linalg.lstsq(X, Y_m, rcond=None)[0]#regression of PCs of images on single units
    return hat_beta

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

sub_sample = 1
#R^2 for each neuron and PC
#for rec in tqdm(range(len(fns))):
#for rec in [-1,]:
rec = 0
da = xr.open_dataset(fns[rec])
fn = fns[rec].split('/')[-1].split('.')[0]
#now load the saved u_r_stim to get the linearity of the eigenmodes
u_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy')
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']
left, right = rf_pos_labs[fn]
imgs = imgs[:60, left:right, da.coords['stim'].values]


# Example usage:
img_size = 60
scales = [1, 0.5, 0.25]
n_thetas = 4
thetas = np.linspace(0, 180 - 180 / n_thetas, n_thetas)
thetas = np.deg2rad(thetas)
inds = []
filters = []
for scale in tqdm(scales):
    sigma = scale * img_size / 4
    f = 1 / (scale * img_size / 2)
    steps_x = steps_y =  np.linspace(scale*img_size/2, img_size - scale*img_size/2, int(1/scale)) - img_size/2
    #increase number of steps so that there is half overlap
    if scale<1:
        steps_x = steps_y =  np.linspace(scale*img_size/2, img_size - scale*img_size/2, int(1.25/scale)) - img_size/2
    for step_x in steps_x:
        for step_y in steps_y:
            for theta in (thetas):
                for phase in [0, np.pi/2]:
                    gabor_filter = create_2d_gabor(img_size, sigma, f, theta, phase, center_x=step_x, center_y=step_y)
                    filters.append(gabor_filter)
                    inds.append(np.array([scale, step_x, step_y, theta, phase,]))
filters_gabor = np.array(filters)
inds = np.array(inds)
#%%
#lets get the betas for the neuron responses and the eigenmode responses in a gabor basis
filter_gab_resp = np.einsum('ijk,jkl->il', filters_gabor, imgs)
filter_gab_resp = np.concatenate((imgs.mean((0,1))[None], filter_gab_resp), 0)
filter_gab_squared_resp = filter_gab_resp[1:]**2
#now concatenate the original and squared features
filter_resp = np.concatenate((filter_gab_resp, filter_gab_squared_resp), 0)
#%%
#normalize variance of each feature
filter_resp = (filter_resp)/filter_resp.std(1, keepdims=True)
X = filter_resp.T
beta = get_beta(X, u_r_stim[:, :10][None])
# %%
sub_ind = len(inds)
plt.plot(beta[:,1])
#print the variance of using the first 648 features vs the last
print('variance of linear features: ', (X[...,:sub_ind] @ beta[:sub_ind, 1]).var())
print('variance of squared features: ', (X[...,sub_ind:] @ beta[sub_ind:, 1]).var())
# %%
#print shapes of beta and X
plt.figure(figsize=(10, 3))
print(beta.shape, X.shape)
i=1
#scatter of of linear and squared features
hat_u = X[..., :sub_ind] @ beta[:sub_ind, i]
plt.subplot(131)
plt.scatter(hat_u, u_r_stim[:, i], s=1, label='linear')
plt.legend()
hat_u = X[..., sub_ind:] @ beta[sub_ind:, i]
plt.subplot(132)
plt.scatter(hat_u, u_r_stim[:, i], s=1, label='squared')
plt.legend()
#use both linear and squared features
hat_u = X @ beta[:, i]
plt.subplot(133)
plt.scatter(hat_u, u_r_stim[:, i], s=1, label='both')
plt.legend()
plt.tight_layout()
# %%
plt.plot(filter_resp.mean(0))
# %%
filter_gab_resp = np.einsum('ijk,jkl->il', filters_gabor, imgs)
filter_gab_resp = np.concatenate((imgs.mean((0,1))[None], filter_gab_resp), 0)
filter_gab_squared_resp = filter_gab_resp**2
filter_gab_squared_resp[0,:] = filter_gab_resp[0,:]
#now concatenate the mean onto filter
#now plot the scatter of just regressing on linear and then squared features
plt.figure(figsize=(10, 3))
i = 1
beta_linear = get_beta(filter_gab_resp.T, u_r_stim[:, i][None])
hat_u = filter_gab_resp.T @ beta_linear
plt.subplot(131)
plt.scatter(hat_u, u_r_stim[:, i], s=1, label='linear')
plt.legend()
beta_squared = get_beta(filter_gab_squared_resp.T, u_r_stim[:, i][None])
hat_u = filter_gab_squared_resp.T @ beta_squared
plt.subplot(132)
plt.scatter(hat_u, u_r_stim[:, i], s=1, label='squared')
plt.legend()
#now both
both = np.concatenate((filter_gab_resp, filter_gab_squared_resp[1:]), 0).T[:,:]
beta_both = get_beta(both, u_r_stim[:, i][None])
hat_u = both @ beta_both
plt.subplot(133)
plt.scatter(hat_u, u_r_stim[:, i], s=1, label='both')
plt.legend()
plt.tight_layout()
#
# %%
#get r2 for just linear, just squared, and both
r2_linear = noise_corr_R2(filter_gab_resp.T, u_r_stim[:, :10][None])[1]
r2_squared = noise_corr_R2(filter_gab_squared_resp.T, u_r_stim[:, :10][None])[1]
r2_both = noise_corr_R2(both, u_r_stim[:, :10][None])[1]
#now plot
res = [r2_linear, r2_squared, r2_both]
for i in range(3):
    plt.plot(res[i], label=['linear', 'squared', 'both'][i])
plt.legend()


# %%
