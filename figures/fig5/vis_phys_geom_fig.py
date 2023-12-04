#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
import scipy.stats as stats

data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'

#%% 
#example plots 
fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
fn = 'ms_natimg2800_M170717_MP033_2017-08-20'#example recording
fn = fn.split('/')[-1].split('.')[0]
neur_snr = np.load(eig_tuning_dir + 'neur_snr_' + fn + '.npy')[0]#TODO fix this
pc_snr = np.load(eig_tuning_dir + 'pc_snr_' + fn + '.npy')[:, 0]
r2s_pc = np.load(eig_tuning_dir + 'lin_r2_pc_' + fn + '.npy')
r2s_neur = np.load(eig_tuning_dir + 'lin_r2_neur' + fn + '.npy')

#%% fig 5a
plt.figure(figsize=(3,2))
plt.plot(range(1,1+len(pc_snr)), pc_snr, c='grey', label='Eigenvector (lower bound)')

plt.plot(range(1, 1+len(neur_snr)), [np.nanmean(neur_snr),]*len(neur_snr), c='k', 
                            label='Neuron average (noise corrected)')
# plot the 95% quantiles of neuron snr
plt.fill_between(range(1, 1+len(neur_snr)), np.nanquantile(neur_snr, 0.025, axis=0), 
                np.nanquantile(neur_snr, 0.975, axis=0), color='k', alpha=0.2, label='Neuron 95% quantile')
        
plt.semilogx()
plt.legend(loc=(1.05, 0.05), framealpha=1, fontsize=8)
plt.xlabel('Eigenvector index')
plt.ylabel('SNR') 
# small annotation of the recording label
#plt.annotate(fn.split('_')[3] + '_' + fn.split('_')[4], (0, -0.2), xycoords='axes fraction', fontsize=8)
plt.savefig('./snr_' + fn + '.pdf', bbox_inches='tight', transparent=True)


#%% fig 5b
plt.figure(figsize=(3,2))
plt.plot(range(1,1+len(r2s_pc)), r2s_pc, c='grey', label='Eigenvector',)
plt.plot(range(1, 1+len(r2s_neur)), [np.nanmean(r2s_neur),]*len(r2s_neur), c='k', 
            label='Single neuron average')
# plot the 95% quantiles of neuron snr
plt.fill_between(range(1, 1+len(r2s_neur)), np.nanquantile(r2s_neur, 0.025, axis=0), 
                np.nanquantile(r2s_neur, 0.975, axis=0), color='k', alpha=0.2, label='Single neuron 95% quantile')
plt.grid()
plt.semilogx()
plt.ylim(-0.1,1)
plt.legend(title='Fraction linear variance', loc=(1.05, 0.05), framealpha=1)
plt.xlabel('Eigenvector index')
plt.ylabel(r'Linearity $(R^2)$') 
plt.annotate(fn.split('_')[3] + '_' + fn.split('_')[4], (0, -0.2), xycoords='axes fraction', fontsize=8)
plt.savefig('./r2_' + fn + '.pdf', bbox_inches='tight', transparent=True)

# %%

#%% fig 5e 
#fraction significant increase in same sign loadings across recordings 
# for the first 20 eigenvectors 
plt.figure(figsize=(3,1.5))
tagged_fns = ['ms_natimg2800_M170714_MP032_2017-08-07',
              'ms_natimg2800_M170604_MP031_2017-06-28', 
             'ms_natimg2800_M170714_MP032_2017-09-14', ]#these are the files with tdtomato tags
crit_p = 0.001
n_eig_vecs = 20
inds = [[1,21], [22, 42], [43, 63]]
for k, fn in enumerate(tagged_fns):
    red_cell = pd.read_csv(data_dir + 'processed_data/red_cell/' + fn[3:] + '.csv', index_col=0)
    red_cell = red_cell.values.squeeze().astype(bool)
    v_r = np.load(eig_tuning_dir + 'sp_cov_neur_u_r_' + fn + '.npy', )
    for i in range(v_r.shape[1]):#set all weights to be majority positive (sign is arbitrary)
        if np.sum(v_r[:, i]>0) < np.sum(v_r[:, i]<0):
            v_r[:, i] *= -1
    is_neg_red = (v_r[red_cell, :n_eig_vecs]<0)
    is_neg_nonred = (v_r[~red_cell, :n_eig_vecs]<0)
    var_red = np.var(is_neg_red, 0)/sum(red_cell)
    var_nonred = np.var(is_neg_nonred, 0)/sum(~red_cell)
    prop_is_neg_red = np.mean(is_neg_red, 0)
    prop_is_neg_nonred = np.mean(is_neg_nonred, 0)
    diff_prop = (prop_is_neg_nonred - prop_is_neg_red)
    abs_dist_half = np.abs(prop_is_neg_red-0.5) - np.abs(prop_is_neg_nonred-0.5)
    t = diff_prop/np.sqrt(var_red + var_nonred)
    p = 2*(1 - stats.norm.cdf(np.abs(t)))
    ind = inds[k]
    plt.scatter(range(ind[0], ind[1]), prop_is_neg_red, c='gray', marker='.')
    plt.scatter(range(ind[0], ind[1]), prop_is_neg_nonred, c='k', marker='.')
    for i, j in enumerate(range(ind[0], ind[1])):
        plt.plot([j, j], [prop_is_neg_red[i], prop_is_neg_nonred[i]], color='k', alpha=0.2)
    crit = (p<crit_p)
    plt.scatter(np.where(crit)[0] + ind[0], prop_is_neg_red[crit],color='green', marker='x', zorder=10)


rec_dates = ['2017-09-14', '2017-06-28', '2017-08-07']
plt.xticks([0, 21, 42,], rec_dates, rotation=0, ha='left', fontsize=8)
plt.grid()
plt.ylim(0,1)
plt.xlabel('Recording date')
plt.ylabel('Fraction negative loadings')
plt.savefig('pc_loading_diff_top_20_all.pdf', bbox_inches='tight', transparent=True, dpi=1000)


# %% Fig 5 e histogram of neuron loadings
tagged_fns = -1
red_cell = pd.read_csv(data_dir + 'processed_data/red_cell/' + fn[3:] + '.csv', index_col=0)
red_cell = red_cell.values.squeeze().astype(bool)
v_r = np.load(eig_tuning_dir + 'sp_cov_neur_u_r_' + fn + '.npy', )
for i in range(v_r.shape[1]):#set all weights to be majority positive (sign is arbitrary)
    if np.sum(v_r[:, i]>0) < np.sum(v_r[:, i]<0):
        v_r[:, i] *= -1
plt.figure(figsize=(3,1.5))
k = 1
bins = 25
plt.hist(v_r[red_cell, k], density=True,  alpha=1, cumulative=False,
                bins=bins,  label='tdtomato+', color='gray')
plt.hist(v_r[~red_cell, k], density=True, cumulative=False,
                alpha=1, bins=bins, histtype='step', label='tdtomato-', color='k')

plt.legend(loc='upper left', fontsize=8)
lim = np.max(np.abs(plt.gca().get_xlim()))
lim = np.max(np.abs(v_r[:, k]))
plt.xlim(-lim, lim)
plt.ylim(-5, None)
plt.grid()
plt.xlabel('Eigenmode neural loading')
plt.ylabel('Density')
plt.title('Eigenmode ' + str(k+1))
plt.savefig('rec_' + fn +' pc_' + str(k) + '_loading_diff' + fn + '.pdf', bbox_inches='tight', transparent=True)

# %% fig 5 d linearity simulations
def create_cov(n, alpha):
    #n is number of filters
    A = np.random.randn(n,n)#create random covariance matrix
    cov =  A @ A.T
    u = np.linalg.svd(cov, full_matrices=False)[0]#reconstruct it with desired eigenvalues
    eig = np.arange(1, n+1)**(-alpha)
    cov = u @ np.diag(eig) @ u.T
    return cov
def run_linearity_sim(n_filters, stim_dim, n_stim, alpha, shift=0):
    stim = np.random.randn(n_stim, stim_dim)
    # creat filters with power-law to adjust correlation of filters
    filter_cov = create_cov(n_filters, alpha)
    #filter weights have filter_cov as covariance
    filters = np.random.multivariate_normal(np.zeros(n_filters), filter_cov, size=(stim_dim ))
    s_resp = stim @ filters #get raw responses
    s_resp = np.maximum(s_resp+shift, 0)#rectify
    s_resp = s_resp - s_resp.mean(0)#subtract mean
    return s_resp, stim

n_filters = 500
stim_dim = 800
n_stim = 5000
alpha = 0.7
s_resp, stim = run_linearity_sim(n_filters, stim_dim, n_stim, alpha, shift=-1.5)

rss = np.linalg.lstsq(stim, s_resp, rcond=None)[1]#get least squares fit of stim to neurons
M = n_stim
D = stim_dim
rss = rss/(M-D)
var_u_r = s_resp.var(0)
r_neuron = 1 - rss/var_u_r
u_r, s_r, v_r = np.linalg.svd(s_resp, full_matrices=False) #PCA of responses
rss = np.linalg.lstsq(stim, u_r, rcond=None)[1]#get least squares fit of stim to PCs
rss = rss/(M-D)
var_u_r = u_r.var(0)
r_pc = 1 - rss/var_u_r
# fig 5 d
plt.figure(figsize=(4,3))
plt.plot(r_pc, label='Eigenmode', color='gray')
plt.plot(r_neuron, label='Neuron', color='black',)
plt.legend()
plt.ylim(0,1)
plt.xlabel('Eigenmode rank')
plt.ylabel('Linearity ' r'($R^2$)')
plt.savefig('linearity_sim.pdf', bbox_inches='tight', transparent=True)
