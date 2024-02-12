#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 17:43:33 2022

@author: dean
"""
#%%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import scipy.io as sio
data_dir = '../../data/stringer_2019/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
#receptive field chosen by eye (see below for bounding boxes on RF estimates)
rf_coords = {'ms_natimg2800_M160825_MP027_2016-12-14':[(25, 65), (35, 85)],
             'ms_natimg2800_M161025_MP030_2017-05-29':[(25, 65), (95, 145)],
            'ms_natimg2800_M170604_MP031_2017-06-28':[(25, 65), (90, 135)],
            'ms_natimg2800_M170714_MP032_2017-08-07':[(20, 55), (20, 60)],
            'ms_natimg2800_M170714_MP032_2017-09-14':[(10, 45), (90, 130)],
            'ms_natimg2800_M170717_MP033_2017-08-20':[(5, 60), (90, 150)],
            'ms_natimg2800_M170717_MP034_2017-09-11':[(20, 60), (90, 135)]}
#%% load neural responses and images
fns = [resp_data_dir + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
fn = 'ms_natimg2800_M160825_MP027_2016-12-14'
da = xr.open_dataset(resp_data_dir + fn + '.nc')
r = da['resp']
n_rep, n_stim, n_neuron,  = r.shape
r = r - r.min()
r = r - r.mean('stim')
xy = da['cellpos']
imgs = sio.loadmat(orig_data_dir + 'images_natimg2800_all.mat')['imgs']

#%%fig 3a example cell locations
plt.figure(figsize=(2.5,2.5), dpi=300)
sub_samp = 10
n = xy[::sub_samp, 0].shape[0]
print(n, ' neurons shown of ', xy.shape[0])
#plt.title('Neurons '+ str(n) + ' of ' + str(xy.shape[0]) + ' shown')
plt.scatter(xy[::sub_samp, 1], xy[::sub_samp, 0], marker='.', s=2, c='w')
plt.axis('square')
plt.xticks([]);plt.yticks([])
#change background to black
plt.gca().set_facecolor('k')#set background to black
plt.annotate(fn, xy=(-0.05, -0.05), xycoords='axes fraction', fontsize=3)
plt.savefig('fig3a_example_cell_locs.png', 
                                bbox_inches='tight', transparent=False, dpi=300)
### fig 3b example response two repeats
sub_samp = 100
r_sub = r[:, ::sub_samp, ::sub_samp]
_, n_neuron_sub, n_stim_sub = r_sub.shape
r_sub_norm = r_sub - r_sub.mean('stim')
r_sub_norm = r_sub/r_sub.mean('rep').std('stim')#zcore across stim for visibility
lim = np.max(np.abs(r_sub_norm))

plt.figure(figsize=(3,4), dpi=300)
plt.subplot(121)
plt.imshow(r_sub_norm[0].T, cmap='coolwarm', vmin=-lim, vmax=lim)
plt.xticks([]);plt.yticks([])
plt.ylabel('Neuron (' + str(n_neuron_sub) + ' of ' + str(n_neuron) + ' shown)')
plt.xlabel('Stimulus\n(' + str(n_stim_sub) + ' of ' + str(n_stim)+ ' shown)')
plt.title('Repeat 1')

plt.subplot(122)
plt.title('Repeat 2')
plt.xticks([]);plt.yticks([])
plt.imshow(r_sub_norm[1].T, cmap='coolwarm',  vmin=-lim, vmax=lim)
plt.colorbar()
plt.annotate(fn, xy=(-0.05, -0.05), xycoords='axes fraction', fontsize=3)
plt.savefig('fig3b_example_resp.pdf', 
                                bbox_inches='tight', transparent=True)
    
# fig 3c example image.
plt.figure(figsize=(4,4))
plt.imshow(imgs[:, :, 3], cmap='gray')#index 3 has a cool bird.
plt.xticks([]);plt.yticks([])
plt.savefig('fig3c_example_im.pdf', bbox_inches='tight', transparent=True)
#%%
# fig 3d population rf

rf = np.load(eig_tuning_dir + 'neur_rf_est_' + fn + '.npy')
rf = (rf**2).sum(-1)
plt.figure()
(r1, r2), (c1, c2) = rf_coords[fn]
plt.plot([c1, c1], [r1, r2], c='r', ls='--')
plt.plot([c2, c2], [r1, r2], c='r', ls='--')
plt.plot([c1, c2], [r1, r1], c='r', ls='--')
plt.plot([c1, c2], [r2, r2], c='r', ls='--')
plt.imshow(rf, cmap='Greys_r')
plt.annotate(fn, xy=(0, -0.05), xycoords='axes fraction', fontsize=3)
plt.xticks([])
plt.yticks([])
plt.savefig('rf_example.pdf', bbox_inches='tight', transparent=True)
# %% example rf coords not in paper but just for sense of 
#how RF limits were chosen

for rec in range(7):
    fn = fns[rec].split('/')[-1].split('.')[0]
    neur_rf = np.load(eig_tuning_dir + 'neur_rf_est_' + fn + '.npy')
    plt.figure()
    rf = (neur_rf**2).sum(-1)
    #set a threshold for the RF greater than 10th quantile
    rf[rf<np.quantile(rf, 0.95)] = 0
    plt.imshow(rf, cmap='Greys_r')
    plt.title(fn)
    n_row, n_col, n_neur = neur_rf.shape
    ax = plt.gca()
    ax.set_xticks(np.arange(0, n_col, 10), fontsize=6)
    ax.set_yticks(np.arange(0, n_row, 10), fontsize=6)
    # set only every other tick label
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    for label in ax.yaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    #plot the bounding box where 
    (r1, r2), (c1, c2) = rf_coords[fn]
    plt.plot([c1, c1], [r1, r2], c='r', ls='--')
    plt.plot([c2, c2], [r1, r2], c='r', ls='--')
    plt.plot([c1, c2], [r1, r1], c='r', ls='--')
    plt.plot([c1, c2], [r2, r2], c='r', ls='--')
    plt.show()
# %%
