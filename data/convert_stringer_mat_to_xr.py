#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 10:52:48 2022

@author: dean
"""
# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from scipy.sparse.linalg import eigsh
import os
import xarray as xr
from tqdm import tqdm
import pandas as pd
with open('./data_dir.txt', 'r') as file:
    data_dir = file.read()
#check if processed_data folder exists, if not make it and all the other data folders
if not os.path.exists(data_dir + '/processed_data/'):
    os.makedirs(data_dir + '/processed_data/')
    os.makedirs(data_dir + '/processed_data/neural_responses/')
    os.makedirs(data_dir + '/processed_data/red_cell/')
    os.makedirs(data_dir + '/processed_data/eig_tuning/')
    os.makedirs(data_dir + '/processed_data/stringer_sn_covs/')
fns = [fn for fn in os.listdir(data_dir + 'orig_stringer2019_data/') if 'natimg2800_' in fn and not 'image' in fn]

for fn in (fns[:]):
    print(fn)
    dat = io.loadmat(data_dir + fn)
    #this saves the tdtomato cell ids for recordings that have them 
    try:
        red_cell = [float(dat['stat'][i]['redcell']) for i in range(len(dat['stat']))]
        #make red cell folder if there is none
        if not os.path.exists(data_dir + '/processed_data/red_cell/'):
            os.makedirs(data_dir + '/processed_data/red_cell/') 
        #save red cell as csv
        pd.DataFrame(red_cell).to_csv(data_dir + '/processed_data/red_cell/' + fn[:-4] + '.csv')

    except:
        print('d')

#this save the responses as xarray, three ways to do it
#raw_ is just the raw responses
#ms_ is the mean subtracted responses
#spont_sub_ is the mean subtracted responses with spontaneous activity subtracted
for rec_trans in ['raw_', 'ms_', 'spont_sub_'][1:2]:
    for fn in (fns[:]):
        print(fn)
        dat = io.loadmat(data_dir + fn)
        resp = dat['stim'][0]['resp'][0] # stim x neurons
        spont = dat['stim'][0]['spont'][0] # timepts x neurons
        istim = (dat['stim'][0]['istim'][0]).astype(np.int32) # stim ids

        istim -= 1 # get out of MATLAB convention
        istim = istim[:,0]
        nimg = istim.max() # these are blank stims (exclude them)
        resp = resp[istim<nimg, :]
        istim = istim[istim<nimg]
        print(resp.shape)

        if rec_trans == 'spont_sub_':
            # subtract spont (32D)
            mu = spont.mean(axis=0)
            sd = spont.std(axis=0) + 1e-6
            resp = (resp - mu) / sd
            spont = (spont - mu) / sd
            sv,u = eigsh(spont.T @ spont, k=32)
            resp = resp - (resp @ u) @ u.T
            resp -= resp.mean(axis=0)#remove average response for each neuron across stimuli
        elif rec_trans == 'ms_':
            resp -= resp.mean(axis=0)
        #otherwise you just keep the raw response

        # split stimuli into two repeats
        NN = resp.shape[1]
        sresp = np.zeros((2, nimg, NN), np.float64)
        inan = np.zeros((nimg,), np.bool)
        img_inds = []
        for n in range(nimg):
            ist = (istim==n).nonzero()[0]#go through all images
            i1 = ist[:int(ist.size/2)][:1]
            i2 = ist[int(ist.size/2):][:1]
            # check if two repeats of stim
            if np.logical_or(i2.size < 1, i1.size < 1):
                inan[n] = 1
            else:
                img_inds.append(n)
                sresp[0, n, :] = resp[i1, :]
                sresp[1, n, :] = resp[i2, :]

        # remove image responses without two repeats
        sresp = sresp[:,~inan,:]
        da = xr.DataArray(sresp, dims=('rep', 'stim', 'unit'),
                     coords=[range(2), img_inds, range(sresp.shape[-1])])

        med = dat['med']  # cell centers (X Y Z)
        pos = xr.DataArray(med,
                           dims=['unit', 'pos'],
                           coords=[range(med.shape[0]), ['x', 'y', 'z']])
        da = xr.Dataset({'resp':da,'cellpos':pos})
        da.to_netcdf(data_dir + '/processed_data/neural_responses/' + rec_trans + fn[:-4] + '.nc'  )
