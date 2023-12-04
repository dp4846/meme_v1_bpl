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
fig_dir = './'
data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'

#%% load neural responses and images
fns = [data_dir + fn for fn in os.listdir(data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
fn ='ms_natimg2800_M170717_MP034_2017-09-11'
da = xr.open_dataset(data_dir + 'xr_conv/' + fn + '.nc')
r = da['resp']
n_rep, n_stim, n_neuron,  = r.shape
r = r - r.min()
r = r - r.mean('stim')
xy = da['cellpos']
imgs = sio.loadmat(data_dir + 'images_natimg2800_all.mat')['imgs']

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
plt.savefig(fig_dir + 'fig3a_example_cell_locs.png', 
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
plt.savefig(fig_dir + 'fig3b_example_resp.pdf', 
                                bbox_inches='tight', transparent=True)
    
# fig 3c example image.
plt.figure(figsize=(4,4))
plt.imshow(imgs[:, :, 3], cmap='gray')#index 3 has a cool bird.
plt.xticks([]);plt.yticks([])
plt.savefig(fig_dir + 'fig3c_example_im.pdf', bbox_inches='tight', transparent=True)

# %% fig 3d population rf
fn = 'ms_natimg2800_M170717_MP034_2017-09-11'
fn = fn.split('/')[-1].split('.')[0]
rf = np.load(data_dir + '/eig_tuning/rf_est_' + fn + '.npy', )
rf = rf/(rf**2).sum()**0.5
plt.figure()
plt.imshow((rf**2)[:10].mean(0), cmap='Greys_r')
plt.annotate(fn, xy=(0, -0.05), xycoords='axes fraction', fontsize=3)
plt.xticks([])
plt.yticks([])
plt.savefig('rf_example.pdf', bbox_inches='tight', transparent=True)
# %%
