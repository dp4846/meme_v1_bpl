#%%

import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import pandas as pd
import scipy.stats as stats
import xarray as xr
data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'

#%% 
#example plots 
fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
fn = 'ms_natimg2800_M170717_MP033_2017-08-20'#example recording
fn = fns[5]
fn = fn.split('/')[-1].split('.')[0]
neur_snr = np.load(eig_tuning_dir + 'neur_snr_' + fn + '.npy')[1]#TODO fix this
pc_snr = np.load(eig_tuning_dir + 'pc_snr_' + fn + '.npy')[1]
r2s_pc = np.load(eig_tuning_dir + 'lin_r2_pc_' + fn + '.npy')
r2s_neur = np.load(eig_tuning_dir + 'lin_r2_neur' + fn + '.npy')
thresh=0.1
ind = neur_snr>thresh
neur_snr_thresh = neur_snr[ind]
r2s_neur_thresh = r2s_neur[ind]

#%% fig 5a
plt.figure(figsize=(4,2))
xticks = [1, 10, 100, 1000]
plt.subplot(121)
plt.plot(range(1,1+len(pc_snr)), pc_snr, c='grey', label='Eigenvector (lower bound)')

plt.plot([0,5000], [np.nanmean(neur_snr),]*2, c='k', 
                            label='Neuron average (noise corrected)')
# plot the 95% quantiles of neuron snr
plt.fill_between(range(1, 5000), np.nanquantile(neur_snr, 0.025, axis=0), 
                np.nanquantile(neur_snr, 0.975, axis=0), color='k', alpha=0.2, label='Neuron 95% quantile')
        
plt.semilogx()
#plt.legend(loc=(1.05, 0.05), framealpha=1, fontsize=8)
#plt.xlabel('Eigenvector index')

plt.ylabel('SNR')
plt.xticks(xticks)
plt.xlim(0, 5000)
plt.xlabel('Eigenvector index')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
# fig 5b
plt.subplot(122)
plt.plot(range(1,1+len(r2s_pc)), r2s_pc, c='grey', label='Eigenvector',)
plt.plot([0,5000], [np.nanmean(r2s_neur_thresh),]*2, c='k', 
                            label='Neuron average (noise corrected)')
# plot the 95% quantiles of neuron snr
plt.fill_between(range(1, 5000), np.nanquantile(r2s_neur_thresh, 0.025, axis=0), 
                np.nanquantile(r2s_neur_thresh, 0.975, axis=0), color='k', alpha=0.2, label='Single neuron 95% quantile')
plt.semilogx()
#remove spines in upper left and righ
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.ylim(-0.1,1)
plt.xticks(xticks)
plt.xlim(0, 5000)
plt.yticks([0,0.25, 0.5,0.75,1])
plt.gca().set_yticklabels(['0', '', '0.5', '', '1'])
plt.gca().set_xticklabels([])
#plt.legend(title='Fraction linear variance', loc=(1.05, 0.05), framealpha=1)
plt.ylabel(r'Linearity $(R^2)$') 
plt.annotate(fn.split('_')[3] + '_' + fn.split('_')[4], (0.01, 0.02), xycoords='axes fraction', fontsize=3)
plt.tight_layout()
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

# %% supplementary snr across recordings fig
fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
#get the ratio of the average snr of top 5 eigenvectors to the average snr of the neurons
pc_snr = []
neur_snr = []
for fn in fns:
    fn = fn.split('/')[-1].split('.')[0]
    neur_snr.append(np.nanmean(np.load(eig_tuning_dir + 'neur_snr_' + fn + '.npy')[1]))
    pc_snr.append(np.mean(np.load(eig_tuning_dir + 'pc_snr_' + fn + '.npy')[1][:10]))
neur_snr = np.array(neur_snr)
pc_snr = np.array(pc_snr)
plt.figure(figsize=(3,1.5))
#plot the ratios
plt.scatter(range(len(neur_snr)), pc_snr/neur_snr, c='k', marker='.')
plt.ylim(0, None)
fn_labels = [fn.split('_')[-2] + '_' + fn.split('_')[-1].split('.')[0] for fn in fns]
plt.xticks(range(len(fn_labels)), fn_labels, rotation=90, ha='left', fontsize=8)
plt.ylabel('Ratio of top 10 PCs \n to single neuron \n average SNR')
plt.xlabel('Recording')
#% print the average of the ratios
print(np.mean(pc_snr/neur_snr))
print(np.min(pc_snr/neur_snr))
# %%
#plot the pc_snrs for all recordings on the same plot wiht semilogx
plt.figure(figsize=(3,1.5))
pc_snrs = []
for fn in fns:
    fn = fn.split('/')[-1].split('.')[0]
    pc_snr = np.load(eig_tuning_dir + 'pc_snr_' + fn + '.npy')[1]
    pc_snrs.append(pc_snr[:1000])
    plt.plot(range(1,1+len(pc_snr)), pc_snr, c='gray', alpha=0.5)
plt.semilogx()
plt.xlabel('Eigenvector index')
plt.ylabel('SNR')
pc_snrs = np.array(pc_snrs)
plt.plot(range(1,1001), np.mean(pc_snrs, 0), c='k')
# %% supplementary figures for fig 5
# performance for truncated and whole image is similar.
gabor_r2 = xr.open_dataarray('eig_gabor_r2.nc')
pc_r2 = xr.open_dataarray('eig_pc_r2.nc')

plt.figure(figsize=(4,2))
dim = pc_r2.coords['dim']
plt.plot(dim, pc_r2.sel(rf_type='truncated', var_stab=False).mean(('pc', 'rec')),
                label='Cropped image', marker='.')
plt.plot(dim, pc_r2.sel(rf_type='whole', var_stab=False).mean(('pc', 'rec')), 
                label='Whole image', marker='.', c='orange')
plt.ylim(0,0.25)
plt.semilogx()
plt.xlim(1,1e3)
plt.xlabel('Number of image PCs')
plt.ylabel(r'Fraction linear variance $(R^2)$')
#set regular ticks at dims
plt.xticks(dim.values, dim.values, rotation=0)
#remove small ticks
plt.gca().tick_params(which='minor', length=0)
plt.legend()
plt.title('Avg. top ten eigenmode linearity \n across recordings')
plt.savefig('pc_r2_saturation.pdf', bbox_inches='tight', transparent=True)

#%% plot of gabor vs pc performance across eigenmodes and recordings.
gabor_r2 = xr.open_dataarray('eig_gabor_r2.nc')
pc_r2 = xr.open_dataarray('eig_pc_r2.nc')
pc_r2 = pc_r2.sel(dim=512, rf_type='truncated', var_stab=False)
gabor_r2 = gabor_r2.sel(var_stab=False)
gabor_r2 = gabor_r2.squeeze().drop('var_stab')
pc_r2 = pc_r2.squeeze().drop(('dim','var_stab', 'rf_type'))
r2 = xr.concat((gabor_r2, pc_r2), 'model')
r2.coords['model'] = ['Gabor', 'PC']

fig, ax = plt.subplots(2,4, figsize=(8,4),)
ymax = 1.0
for i, rec in enumerate(r2.coords['rec'].values):
    ax[i//4, i%4].plot(r2.coords['pc']+1, r2.sel(rec=rec, model='Gabor'), marker='.', label='Gabor filters', c='b', alpha=0.5)
    ax[i//4, i%4].plot(r2.coords['pc']+1, r2.sel(rec=rec, model='PC'), marker='.', label='Image PCs', c='orange', alpha=0.5)
    ax[i//4, i%4].set_ylim(0, ymax)
    ax[i//4, i%4].set_xticks([1,5,10])
    #set yticks 
    ax[i//4, i%4].set_yticks([0, 0.25, 0.5, 0.75, 1], [0, '', 0.5, '', 1])

    
    ax[i//4, i%4].set_title(rec.split('00_')[-1], fontsize=8)
    if i==0:
        ax[i//4, i%4].set_ylabel(r'$R^2$')
        ax[i//4, i%4].set_xlabel('Eigenmode index')
    else:
        ax[i//4, i%4].set_yticklabels([])
        ax[i//4, i%4].set_xticklabels([])

ax[0,0].legend(loc='upper right', framealpha=1, fontsize=10)
#remove ticks and labels
ax[-1,-1].set_xticks([])
ax[-1,-1].set_yticks([])
#remove spines all
ax[-1,-1].spines['top'].set_visible(False)
ax[-1,-1].spines['right'].set_visible(False)
ax[-1,-1].spines['bottom'].set_visible(False)
ax[-1,-1].spines['left'].set_visible(False)

plt.tight_layout()
plt.savefig('gabor_compared_to_pc.pdf', bbox_inches='tight', transparent=True)
# %% get max difference between gabor and pc
diff = r2.sel(model='Gabor') - r2.sel(model='PC')
print(diff.max())