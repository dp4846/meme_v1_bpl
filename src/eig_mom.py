#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:02:46 2021

@author: dean
"""

import numpy as np
from scipy import linalg as LA
from scipy.optimize import least_squares
from tqdm import tqdm
from math import comb

### simulation functions
def sig_noise_mats(d, n_eig, s_eig, align=False):
    """
    function to generate signal and noise matrices with desired eigenspectrums with option to align eigenvectors
    used to multiply by random observations (cov=I, E=0) to generate data with desired eigenspectrums
    Parameters
    ----------
    d : int
        number of neurons.
    n_eig : array
        noise eigenspectrum.
    s_eig : array
        signal eigenspectrum.
    align : bool, optional
        whether noise and signal eigenvectors are the same or independently chosen. The default is False.

    Returns
    -------
    S : array
        dxd signal matrix.
    N : array
        dxd noise matrix.
    """
    #d is number of neurons, n_eig is noise eigenspectrum, s_eig is signal eigenspectrum, 
    # align is whether noise and signal eigenvectors are the same or independently chosen
    # returns a rep X stimuli X neuron matrix with desired eigenspectrums
    S = np.random.normal(size=(d, d))#randomly initialize signal matrix
    u, s, vh = np.linalg.svd(S, full_matrices=False,)#svd on random signal matrix
    if align:#eigenvectors of signal and noise are the same
        S = u @ np.diag(s_eig) @ vh#force eigenspectrum on signal matrix
        N = u @ np.diag(n_eig) @ vh#force eigenspectrum on noise matrix, with same eigenvecs as signal
    else:
        S = u @ np.diag(s_eig) @ vh#force eigenspectrum of signal
        N = np.random.normal(size=(d, d))#independent noise matrix
        u, s, vh = np.linalg.svd(N, full_matrices=False,)#svd on noise matrix
        N = u @ np.diag(n_eig) @ vh#force eigenspectrum of noise
    return S, N

def sig_noise_samples(n, n_rep, S, N):
    #n is number of stimuli, n_rep is number of reps, S is signal matrix, N is noise matrix
    #S and N are d X d matrices
    d = S.shape[0]
    X = np.random.normal(size=(n, d))#random observations by neurons matrix with mean 0 and variance 1
    Y = X@S #generate signal observations
    N_r = np.array([np.random.normal(size=(n, d)) @ N for i in range(n_rep)]) #generate trial-to-trial noise observations
    Y_r = Y[np.newaxis] + N_r#gives rep X stimuli X neuron tensor
    return Y_r, N_r, Y #return signal plus noise observations, noise observations, and signal observations

def get_power_law_SN_sim(c_sig, c_noise, alpha_sig, alpha_noise, d_neurons,
                         n_stim, n_rep, align, return_params=False):
    """

    Parameters
    ----------
    c_sig : float
        signal variance intercept. 
    c_noise : float
        noise variance intercept.
    alpha_sig : float
        signal variance power law exponent.
    alpha_noise : float 
        noise variance power law exponent.
    d_neurons : int
        number of neurons.
    n_stim : int
        number of stimuli.
    n_rep : int
        number of repetitions.
    align : bool, optional
        whether noise and signal eigenvectors are the same or independently chosen. The default is False.

    Returns
    -------
    Y_r : array
        rep X stimuli X neuron tensor of signal plus noise observations.
    ind : array
        array of indices for eigenspectrum.
    s_eig : array
        array of signal eigenspectrum.
    n_eig : array
        array of noise eigenspectrum.
    """

    ind = np.arange(1, d_neurons+1).astype(float)#indices for eigenspectrum
    s_eig = (c_sig * ind**(-alpha_sig))#signal eigenspectrum
    n_eig = (c_noise * ind**(-alpha_noise))#noise  eigenspectrum
    S, N = sig_noise_mats(d_neurons, n_eig**0.5, s_eig**0.5, align)#signal and noise matrices, sqrt because S^TS will give squared eigenvalues
    Y_r, N_r, Y = sig_noise_samples(n_stim, n_rep, S, N)#signal plus noise observations, noise observations, and signal observations
    if return_params:
        return Y_r, ind, s_eig, n_eig
    else:
        return Y_r

def get_broken_power_law_SN_sim(c_sig, c_noise, 
                                alphas_sig, alphas_noise, 
                                break_points_sig, break_points_noise, 
                                d_neurons, n_stim, n_rep, align, return_params=False):
    
    ind = np.arange(1, d_neurons+1).astype(float)#indices for eigenspectrum
    s_eig = np.exp(get_bpl_func_all(A=alphas_sig, log_c1=c_sig, B=break_points_sig, ind=ind))
    n_eig = np.exp(get_bpl_func_all(A=alphas_noise, log_c1=c_noise, B=break_points_noise, ind=ind))
    S, N = sig_noise_mats(d_neurons, n_eig**0.5, s_eig**0.5, align)#signal and noise matrices, sqrt because S^TS will give squared eigenvalues
    Y_r, N_r, Y = sig_noise_samples(n_stim, n_rep, S, N)#signal plus noise observations, noise observations, and signal observations
    if return_params:
        return Y_r, ind, s_eig, n_eig
    else:
        return Y_r

#eigenmoment based estimator functions

def n_obs_needed_to_whiten_d_dims(d_dims):
    # based on conservative scaling of rule of thumb
    # that you need O(dim log dim) samples to estimate covariance
    # for multivariate normal
    n = int(10*d_dims*np.log(d_dims))
    return n

def rescale_eig_mom(eig_moms, c):
    #if your data was multiplied by c then use this to invert
    return [eig_mom*(c)**(-(i+1)*2.) for i, eig_mom in enumerate(eig_moms)]

def raw_to_central(mu_raw):
    mean = mu_raw[0] # the mean is the first raw moment
    mu_N_cs = np.zeros(len(mu_raw)) # initialize the central moments
    mu_N_cs[0] = mean # the first central moment is not the mean but thats fine here.
    for N in range(2, len(mu_raw) + 1):# go from the variance onwards
        mu_N_c = (((-1)**N)*(mean**N) + #first term is always the mean to the Nth power
                  np.sum([comb(N, j) * (-1)**(N-j) * mu_raw[j-1] * mean**(N-j) 
                  for j in range(1, N+1)]))
        mu_N_cs[N-1] = mu_N_c
    return mu_N_cs

def p_moms(A, k_moms, d_dims):
    # eigenmoment estimator
    # see Algorithm 1 from: 'Spectrum estimation from samples. Kong and Valiant https://arxiv.org/abs/1602.00061
    n = np.shape(A)[0]
    F = np.triu(A,1)
    F_i = np.eye(n)
    H = []
    for i in range(k_moms):
        H.append(np.trace(F_i@A/comb(n, i+1)/d_dims))
        F_i = F_i@F
    return np.array(H)

def signal_er_eig_momk_centered(Y_1, Y_2, k_moms=5):
    #Y_1 and Y_2 are the two repeats of n_stim X n_neur
    #we assume stimuli selection is iid.
    n_stim, n_neur = Y_1.shape
    if n_stim%2>0:
        Y_1 = Y_1[:-1]
        Y_2 = Y_2[:-1]
    # this step sets mean to 0 while preserving covariance
    # see section in paper: 'Extension to noisy data' 2nd paragraph
    norm = 1/np.sqrt(2)
    Y1_sur = (Y_1[::2] - Y_1[1::2])*norm
    Y2_sur = (Y_2[::2] - Y_2[1::2])*norm
    # this stim X stim signal covariance estimate is then unbiased
    A = Y1_sur@Y2_sur.T
    #plugged into the eigenmoment estimator it gives an unbiased estimate of the eigenmoments
    k_eig_mom_ests = p_moms(A, k_moms, n_neur)
    return k_eig_mom_ests

def get_bpl_func_all(A, log_c1, B, ind):
    #A is the list of slopes
    #log_c1 is the intercept of the first powerlaw
    #B is the integer list of break points, one less than length of A
    #ind is the 1 - N+1 list of eigenvalue indices

    log_cs = [log_c1,]
    for i in range(len(B)):
        log_c = (log_cs[i] - A[i] * np.log(B[i]) + A[i+1] * np.log(B[i]))
        log_cs.append(log_c)
    B = [0,] + B + [None,]
    log_func = []
    for i in range(len(log_cs)):
        log_func += list(log_cs[i] + np.log((ind[B[i]:B[i+1]]**(-A[i]))))

    return np.array(log_func)

def eig_moms_from_alpha_all_log_cost(x, args):
    log_c1 = x[0]
    A = x[1:]#all the alphas
    [H, ind, ] = args[:2] #estimated eigmoments, indices of eigenvalues
    B = args[2:]#all of the change points.
    lam = np.exp(get_bpl_func_all(A, log_c1, B, ind))
    log_moms = np.log(np.array([np.mean(lam**(i+1)) for i in range(len(H))]))
    return log_moms - np.log(H)

def eig_moms_from_alpha_all_cost(x, args):
    log_c1 = x[0]
    A = x[1:]#all the alphas

    [H_est, ind, ] = args[:2] #estimated eigmoments, indices of eigenvalues
    B = args[2:]#all of the change points.
    lam = np.exp(get_bpl_func_all(A, log_c1, B, ind))


    H_calc = np.array([np.mean(lam**(i+1)) for i in range(len(H_est))])
    return H_calc - H_est

def eig_moms_from_alpha_all_cost_weighted(x, est_eig_mom, ind, break_points,
                                          transform, W):
    #optimizing parameters
    log_c1 = x[0]
    A = x[1:]#all the alphas
    assert(len(break_points) + 1 == len(A))
    #auxilliary parameters
    lam = np.exp(get_bpl_func_all(A, log_c1, break_points, ind))

    #fit eigenmoments
    fit_eig_moms = np.array([np.mean(lam**(p+1)) for p in range(len(est_eig_mom))])
    fit_eig_moms = eig_mom_transforms(fit_eig_moms, transform=transform)

    r = W @ (fit_eig_moms - est_eig_mom)/len(est_eig_mom)#weighted residuals
    return r

def bs_eig_mom(Y_r, k_moms, n_samps=None):
    if n_samps is None:
        n_samps = n_obs_needed_to_whiten_d_dims(k_moms)
    n_stim= np.shape(Y_r)[1]
    Hs = np.zeros((n_samps, k_moms))
    for i in tqdm(range(n_samps)):
        stim_resamp = np.random.choice(n_stim, replace=True, size=n_stim)
        b_Y_r = Y_r[:, stim_resamp]
        H = signal_er_eig_momk_centered(b_Y_r[0], b_Y_r[1], k_moms=k_moms)
        Hs[i, :] = H
    return Hs

def eig_mom_transforms(eig_mom, transform='raw'):

    if len(eig_mom.shape)!=2:
        eig_mom_trans = eig_mom.copy()[np.newaxis]
    else:
        eig_mom_trans = eig_mom.copy()

    samples, k_moms = eig_mom_trans.shape

    if transform == 'root':
        eig_mom_trans[eig_mom_trans<0] = 0
        eig_mom_trans = eig_mom_trans**(np.arange(1, k_moms + 1)**(-1.))[np.newaxis]
    elif transform == 'log':
        eps = 1e-32
        eig_mom_trans[eig_mom_trans < eps] = eps
        eig_mom_trans = np.log(eig_mom_trans)
    elif transform == 'centered_raw':
        eig_mom_trans = np.array([raw_to_central(eig_mom_trans[i]) for i in range(samples)])

    return eig_mom_trans.squeeze()


def bs_eig_mom_cov_W(Y_r=None, k_moms=None, n_samps=None, return_W=False,
                     return_samps=False, transform='raw', bs_est_eig=None):
    # get bootstrap estimated covariance of eigenmoments
    # and assosciated weighting matrix
    if bs_est_eig is None:
        bs_est_eig = bs_eig_mom(Y_r, k_moms, n_samps=n_samps)

    bs_est_eig = eig_mom_transforms(bs_est_eig, transform=transform)

    bC = np.cov(bs_est_eig.T)
    if return_W:
        u, s, vh = np.linalg.svd(bC)
        W = u @ np.diag(s ** (-0.5)) @ vh
        return np.real(W)
    else:
        return bC

def fit_broken_power_law_meme_W(Y_r, k_moms, break_points, log_c1, slopes,
                                 return_full_res=False, W=None, transform='raw',
                                 bs_est_eig=None, est_eig_mom=None):
    '''
    Parameters
    ----------
    Y_r : xarray.core.dataarray.DataArray
        repeats (2) X # stimuli X # units
    k_moms : int
        the number of eigenmoments on which the powerlaw will be fit, higher
        eigenmoments are noisier.
    break_points : list
         N-1 break points for broken powerlaw
    log_c1 : TYPE, optional
        the log of the value of the first eigenvalue.
    slopes : TYPE, optional
        list of N slopes.
    return_res : boolean, optional
        return the full result of the optimization. The default is False.
    W : array, optional
        a matrix for weighting the eigenmoment errors. The default is None
        and will result in weights being estimated.
    transform : string, optional
        the transformation to apply to the eigenmoments before fitting, can be
        'raw (default), 'root', or 'log'.
    bs_est_eig : array, optional
        bootstrap estimated eigenmoments. The default is None and will result in
        them being estimated. Useful if you have already calculated these.
    Returns
    -------
    res_S : array
        the parameters of the fit. if return_res is False, the parameters are
        returned, otherwise the full result of the optimization is returned. 
    '''
    n_reps, n_stim, d_neurons= np.shape(Y_r)
    ind = np.arange(1, d_neurons+1).astype(float)#index of eigenspectrum
    if W is None:#estimate the weighting matrix if none was provided
        W = bs_eig_mom_cov_W(Y_r, k_moms,
                             return_W=True,
                             transform=transform, 
                             bs_est_eig=bs_est_eig)
    if est_eig_mom is None:
        est_eig_mom = signal_er_eig_momk_centered(Y_r[0], Y_r[1],
                                        k_moms=k_moms)
        est_eig_mom = eig_mom_transforms(est_eig_mom, transform=transform)

    #initial guess at parameters
    x0 = [log_c1, ] + slopes
    bounds=([-np.inf,] + [0,]*(len(x0)-1),#slopes cannot be negative
            [np.inf,]*(len(x0)))
    kwargs = {'est_eig_mom':est_eig_mom,
            'ind':ind,
            'break_points':break_points,
            'transform':transform,
            'W':W}#kwargs wills be passed eig_moms_from_alpha_all_cost_weighted by least_squares
    res_S = least_squares(eig_moms_from_alpha_all_cost_weighted, x0,
                            kwargs = kwargs,
                            bounds = bounds,
                            max_nfev=1000,
                            method='dogbox',
                            x_scale='jac',
                            jac='3-point',
                            ftol=None,
                            gtol=None)

    if return_full_res:
        return res_S
    else:
        return res_S.x


def break_point_search_fit_broken_power_law_meme(Y_r, k_moms, all_break_points, log_c1, slopes,
                                 return_full_res=False, W=None, transform='raw',
                                 bs_est_eig=None, est_eig_mom=None):
    # fit broken power law to eigenmoments trying all break points and choosing that with the lowest cost
    #first get estimated eigenmoments
    if est_eig_mom is None:
        est_eig_mom = signal_er_eig_momk_centered(Y_r[0], Y_r[1],
                                        k_moms=k_moms)
        est_eig_mom = eig_mom_transforms(est_eig_mom, transform=transform)
    else:
        est_eig_mom = eig_mom_transforms(est_eig_mom, transform=transform)

    if W is None:#estimate the weighting matrix if none was provided
        W = bs_eig_mom_cov_W(Y_r, k_moms,
                                return_W=True,
                                transform=transform, 
                                bs_est_eig=bs_est_eig)

    # then fit broken power law for each set of breaks
    fit_results = []
    for break_points in tqdm(all_break_points):
        res  = fit_broken_power_law_meme_W(Y_r, k_moms, break_points, log_c1, slopes,
                                 return_full_res=True, W=W, transform=transform,
                                  est_eig_mom=est_eig_mom)
        fit_results.append(res)
    
    #then find the best fit
    best_fit = np.argmin([res.cost for res in fit_results])
    if return_full_res:#return the full result if requested (useful for debugging, includes success status)
        return [fit_results[best_fit], all_break_points[best_fit], fit_results]
    else:#otherwise just return the parameters
        return [fit_results[best_fit].x, all_break_points[best_fit]]



# covariance estimation 
def get_near_psd(A):
    C = (A + A.T)/2
    eigval, eigvec = LA.eig(C)
    eigval = np.array(eigval.copy())
    eigval[eigval < 0] = 0
    C = eigvec @ np.diag(eigval) @ eigvec.T
    return C

#TODO remove reliance on xarray
def calc_noise_cov(resp):
    n_rep, n_stim, n_neur = resp.shape
    resp_ms = resp - resp.mean('rep')
    #only 2 reps so unbiased estimate is divided by 1
    resp_ms = resp_ms.transpose('stim', 'unit', 'rep',).values
    noise_cov = np.zeros((n_neur, n_neur))
    for i in (range(n_stim)):
        # form each unit X unit 
        noise_cov += resp_ms[i] @ resp_ms[i].T#sum across stimuli of noise cov
    noise_cov = noise_cov/n_stim#averaging cov estimates across stimuli.
    noise_cov = get_near_psd(noise_cov)
    noise_cov = np.real(noise_cov)
    return noise_cov

def calc_sig_cov(resp, force_psd=True):
    n_rep, n_stim, n_neur = resp.shape
    resp_ms = resp - resp.mean('stim')#subtract off mean
    sig_cov = (resp_ms[0].values.T @ resp_ms.values[1])/(n_stim-1)#get sample cov
    if force_psd:
        sig_cov = get_near_psd(sig_cov)
        sig_cov = np.real(sig_cov)
    return sig_cov

### dimensionality estimators
def calc_pr(eig):
    pr = (np.sum(eig)**2)/np.sum(eig**2)
    return pr
def naive_pr(cov):
    eig = np.linalg.eigvals(cov)
    pr = calc_pr(eig)
    return pr

### stringer et al '19 estimators
def cvpca(resp, force_psd=False):
    #2 X stim X neuron
    resp_ms = resp - resp.mean(1, keepdims=True)
    n_stim, n_neur = resp[0].shape

    sp_cov = (resp_ms[0].T @ resp_ms[1])/(n_stim-1)
    if force_psd:
        sp_cov = get_near_psd(sp_cov)
        sp_cov = np.real(sp_cov)

    nosp_cov = (resp_ms[0].T @ resp_ms[0])/(n_stim-1)

    _, eig_vec = LA.eigh(nosp_cov)
    cvpca_eigval = np.array([eig_vec[:, n].T @ sp_cov @ eig_vec[:, n]
                          for n in range(n_neur)])
    return cvpca_eigval[::-1]

# original stringer estimator
#from sklearn.decomposition import PCA
# def cvPCA(X):
#     ''' X is 2 x stimuli x neurons '''
#     pca = PCA(n_components=min(1024, X.shape[1])).fit(X[0].T)
#     u = pca.components_.T
#     sv = pca.singular_values_
    
#     xproj = X[0].T @ (u / sv)
#     cproj0 = X[0] @ xproj
#     cproj1 = X[1] @ xproj
#     ss = (cproj0 * cproj1).sum(axis=0)
#     return ss

def get_powerlaw(eig_est, trange, weight=True):
    log_eig_est = np.log(np.abs(eig_est))
    y = log_eig_est[trange][:,np.newaxis]
    trange += 1
    nt = trange.size
    x = np.concatenate((-np.log(trange)[:,np.newaxis], np.ones((nt,1))), axis=1)

    if weight:
        w = 1.0 / trange.astype(np.float32)[:, np.newaxis]
    else:
        w = 1.0

    b = np.linalg.solve(x.T @ (x * w), (w * x).T @ y).flatten()

    allrange = np.arange(0, eig_est.size).astype(int) + 1
    x = np.concatenate((-np.log(allrange)[:,np.newaxis], np.ones((eig_est.size,1))), axis=1)
    ypred = np.exp((x * b).sum(axis=1))

    return ypred, b

def noise_corr_R2(X, Y, noise_corrected=True):
    #X is observation x feature
    #Y is repeat x observation X neuron
    K, M, N = Y.shape
    M, D = X.shape
    Y_m = Y.mean(0)
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
