#%%
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

import sys
sys.path.append('../../src/')
import eig_mom as em
data_dir = '../../data/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
raw_data_dir =  data_dir + '/orig_stringer2019_data/'
np.random.seed(42)
#%%
fns = [fn for fn in os.listdir(resp_data_dir) if 'natimg2800_' in fn 
            and not 'npy' in fn and 'ms' in fn and '.nc' in fn and '0_M' in fn]

#first make data structures to hold results
n_neurs = []
n_rec = len(fns)
# for holding onto eigenspectra raw results
fn_nms = [fns[rec].split('/')[-1].split('.')[0] for rec in range(n_rec)]
for rec in fn_nms:
    n_neurs.append(xr.open_dataset(resp_data_dir + rec + '.nc')['resp'][..., :].shape)
# %% calculate eigennmoments and bootstrap distribution

moms = []
k_moms = 10
n_bs_samps = em.n_obs_needed_to_whiten_d_dims(k_moms)*3
recs = ['raw_natimg2800_M161025_MP030_2017-05-29', 'ms_natimg2800_M161025_MP030_2017-05-29']
for rec in tqdm(recs):
    ds = xr.open_dataset(resp_data_dir + rec + '.nc')
    resp = ds['resp']
    resp = resp#*1e-3#/(resp[0]*resp[1]).mean('stim').sum('unit')**.5
    #estimate eigenmoments
    mom_est = em.signal_er_eig_momk_centered(resp[0].values, resp[1].values, k_moms)
    #bootstrap eigenmoment distribution
    mom_dist = em.bs_eig_mom(resp.values, k_moms, n_bs_samps)    
    moms.append([mom_est, mom_dist])   

#%%
#first check whether moments are similar for the two methods
plt.plot(np.abs(moms[0][0]-moms[1][0]))
print((moms[0][0]-moms[1][0])[:4])
print(moms[0][0][:4])
print(moms[1][0][:4])
plt.semilogy()
#%%    
Y_r = resp.values
n_rep,  n_stim, n_neur = Y_r.shape
#get the cvpca raw estimates
#%%
cvpca = em.cvpca(Y_r, force_psd=False)

# %% create grid of model parameters
n_neur = resp.shape[-1]
k_moms_fit = 10
n_breaks = 25
ind = np.arange(1, n_neur + 1)
# load  estimated eigenmoments and distribution
est_eig_mom = moms[0][0]
m1 = est_eig_mom[0]
n_log_c1_range = 20
n_slope_range = 20
log_c1_range = np.linspace(np.log(m1), np.log(m1*n_neur), n_log_c1_range)
slope_range = np.linspace(0.001, 2, n_slope_range)
init_break_points = list(range(2, 50, 2))
search_break_points = init_break_points + list(np.logspace(np.log10(50), np.log10(n_neur),
                                n_breaks-len(init_break_points)).astype(int))

slope_list = []
log_c1_list = []
break_point_list = []
#As will be the cartestian product of slopes with itself
for log_c1 in tqdm(log_c1_range):
    for slope_1 in slope_range:
        for slope_2 in slope_range:
            for break_point in search_break_points:
                slope_list.append([slope_1, slope_2])
                log_c1_list.append(log_c1)
                break_point_list.append([break_point,])
                

#%%               
#get the eigenmoments of each parameter set
model_eig_mom = []
for i in tqdm(range(len(slope_list))):
    slopes = slope_list[i]
    log_c1 = log_c1_list[i]
    break_points = break_point_list[i]
    sig = em.get_bpl_func_all(slopes, log_c1, break_points, ind)
    mp = np.array([np.mean(np.exp(sig)**(j+1)) for j in range(k_moms_fit)])
    model_eig_mom.append(mp)
model_eig_mom = np.array(model_eig_mom)
# %%

scale = 1
for i in range(2):

    est_eig_mom = moms[i][0]
    bs_est_eig = moms[i][1]
    # est_eig_mom = np.array(rescale_eig_mom(est_eig_mom, scale))
    # bs_est_eig = np.array([rescale_eig_mom(bs_est_eig[i], scale) for i in range(bs_est_eig.shape[0])])
    model_eig_mom_resc = model_eig_mom#np.array([rescale_eig_mom(model_eig_mom[i], scale) for i in range(model_eig_mom.shape[0])])

    #weighting matrix according to eigenmoment covariance for MEME
    W = em.bs_eig_mom_cov_W(transform='raw', bs_est_eig=bs_est_eig, return_W=True)
    #now get the distance between the model and the estimated eigenmoments
    r = ((W @ (model_eig_mom_resc - est_eig_mom[None]).T)**2).mean(0)#weighted residuals
    min_ind = np.argmin(r)
    min_r = r[min_ind]
    min_slope = slope_list[min_ind]
    min_log_c1 = log_c1_list[min_ind] + np.log(scale)
    min_break_point = break_point_list[min_ind]
    print('')
    print('min_r: ', min_r)
    print('min_slope: ', min_slope)
    print('min_log_c1: ', min_log_c1)
    print('min_break_point: ', min_break_point)
    #plot the best fit
    plt.figure()
    slopes = min_slope
    log_c1 = min_log_c1
    break_points = min_break_point
    sig = em.get_bpl_func_all(slopes, log_c1, break_points, ind)
    mp = np.array([np.mean(np.exp(sig)**(j+1)) for j in range(k_moms_fit)]) 
    plt.plot(range(1, 1 + k_moms_fit), mp, label='model_eig_mom')
    plt.plot(range(1, 1 + k_moms_fit), est_eig_mom, label='est_eig_mom')
    plt.semilogy()
    plt.figure()
    plt.plot(W@(mp-est_eig_mom))
    plt.title('residuals')
    plt.figure()
    # plt.plot(np.sort(r)[:20])
    # plt.title('error')
    # #put parameters on the xtiks
    # plt.gca().set_xticks(range(20))
    # plt.gca().set_xticklabels([str(np.array(slope_list[i]).round(2)) + ' ' + 
    #                             str(np.array(log_c1_list[i]).round(2)) + ' ' + 
    #                             str(np.array(break_point_list[i]).round(2)) for i in np.argsort(r)[:20]], rotation=90)
    #make commented figure but swap x and y
    plt.plot(np.sort(r)[:20],np.arange(20))
    plt.title('error')
    #put parameters on the xtiks
    plt.gca().set_yticks(range(20))
    plt.gca().set_yticklabels([str(np.array(slope_list[i]).round(2)) + ' ' + 
                                str(np.array(log_c1_list[i]).round(2)) + ' ' + 
                                str(np.array(break_point_list[i]).round(2)) for i in np.argsort(r)[:20]], rotation=0)
    plt.show()
    
                                
    # #%%

    slopes = min_slope
    break_points = min_break_point
    init_log_c1 = min_log_c1
    search_break_points_list = [[break_point,] for break_point in search_break_points]
    params, b1 = em.break_point_search_fit_broken_power_law_meme(Y_r, k_moms_fit, 
                                    search_break_points_list, init_log_c1, slopes,
                                    return_full_res=False, W=W, transform='raw',
                                    bs_est_eig=bs_est_eig, est_eig_mom=est_eig_mom)

    log_c1, alpha_1, alpha_2 = params
    slopes = [alpha_1, alpha_2]
    break_points = b1
    sig = em.get_bpl_func_all(slopes, log_c1, break_points, ind)
    mp = np.array([np.mean(np.exp(sig)**(j+1)) for j in range(k_moms_fit)]) 
    #now print error
    print(((W @ (mp - est_eig_mom))**2).mean())
    print('alpha_1: ', alpha_1)
    print('alpha_2: ', alpha_2)
    print('log_c1: ', log_c1)
    print('break_point: ', b1)
    print('')
# %% check on rescale
k_moms = 5
scale = 1e-4
ds = xr.open_dataset(resp_data_dir + rec + '.nc')
resp = ds['resp'].values.astype(np.float64)
resp = resp*scale#/(resp[0]*resp[1]).mean('stim').sum('unit')**.5
#estimate eigenmoments
mom_est_sc = em.signal_er_eig_momk_centered(resp[0], resp[1], k_moms)

ds = xr.open_dataset(resp_data_dir + rec + '.nc')
resp = ds['resp'].values.astype(np.float64)
resp = resp#/(resp[0]*resp[1]).mean('stim').sum('unit')**.5
#estimate eigenmoments
mom_est_unsc = em.signal_er_eig_momk_centered(resp[0], resp[1], k_moms)


def rescale_eig_mom(eig_moms, c):
    #if your data was multiplied by c then this function inverts that effect on the eigenmoments
    return [eig_mom*((c)**(-(i+1)*2.)) for i, eig_mom in enumerate(eig_moms)]
print(np.array(rescale_eig_mom(mom_est_sc, scale)) - mom_est_unsc)



# %%
import numpy as np

# Create a NumPy array with lower precision
arr = np.array([1.234, 2.345, 3.456])

# Increase precision by converting to a higher precision data type
arr_high_precision = arr.astype(np.float64)  # You can choose np.float64 for higher precision

# Print the arrays to see the difference in precision
print("Original array:", arr)
print("Array with increased precision:", arr_high_precision)
# %%
k_moms = 5
scale = 1e-4
ds = xr.open_dataset(resp_data_dir + rec + '.nc')
resp = ds['resp'].values.astype(np.float64)
resp = resp + np.random.randn(*resp.shape)/1e04
resp = resp*scale#/(resp[0]*resp[1]).mean('stim').sum('unit')**.5
#estimate eigenmoments
mom_est_sc = em.signal_er_eig_momk_centered(resp[0], resp[1], k_moms)

resp = resp/scale#/(resp[0]*resp[1]).mean('stim').sum('unit')**.5
#estimate eigenmoments
mom_est_unsc = em.signal_er_eig_momk_centered(resp[0], resp[1], k_moms)


def rescale_eig_mom(eig_moms, c):
    #if your data was multiplied by c then this function inverts that effect on the eigenmoments
    return [eig_mom*(c)**(-(i+1)*2.) for i, eig_mom in enumerate(eig_moms)]
print(np.array(rescale_eig_mom(mom_est_sc, scale)) - mom_est_unsc)
# %%
np.set_printoptions(precision=128, suppress=False)
a = np.float128(3.)**np.float128(-1.)
# Print the array
print(a)
# %%
# Create a NumPy array with higher precision data type
arr = np.array([1.234567890123456789012345678901, 2.345678901234567890123456789012, 3.456789012345678901234567890123])

# Set printing options to display all digits
np.set_printoptions(precision=None, suppress=False)

# Print the array
print(arr)
# %%
