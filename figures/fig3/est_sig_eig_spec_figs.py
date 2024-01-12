#%%
import xarray as xr
import pandas as pd
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from matplotlib import rc
from scipy.stats import chi2
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
from tqdm import tqdm
from scipy.stats import multivariate_normal as mn
sys.path.append('../../src/')
import eig_mom as em
def rescale_eig_mom(eig_moms, c):
    #if your data was divided by c then to get the scaling of the eigenmoments use this function
    return [eig_mom*(c)**(-(i+1)*2.) for i, eig_mom in enumerate(eig_moms)]

with open('../../data/data_dir.txt', 'r') as file:
    data_dir = file.read()
resp_data_dir =  data_dir + 'processed_data/neural_responses/'
ci_dir = './bpl2_sims/'

#get names of all csv files in ci_dir
ci_files = os.listdir(ci_dir)
ci_files = [f for f in ci_files if f[-4:] == '.csv']
fit_df = pd.read_csv('str_pt_estimates.csv').set_index('fn_nms')
#%%
#first make the example plot
i = 2
nm = fit_df.iloc[i].index
nm = 'ms_natimg2800_M170717_MP034_2017-09-11'
ci_file = [f for f in ci_files if nm in f][0]
#read csv in ci_file
n_recs = len(fit_df)
ci_dfs = []
for rec in range(n_recs):
    fn_nm = fit_df['file_name'][rec]
    #get ci_file in ci_files with nm in it
    ci_file = [f for f in ci_files if fn_nm in f][0]
    #read csv in ci_file
    ci_df = pd.read_csv(ci_dir + ci_file)
    ci_dfs.append(ci_df)
ci_df = pd.concat(ci_dfs, keys=fit_df.index)
# load scaling
#get raw data file with nm and get the number of n_neur
resp_data_file = [f for f in os.listdir(resp_data_dir) if nm in f][0]
raw_data = xr.open_dataset(resp_data_dir + resp_data_file)['resp']
n_neur = raw_data.coords['unit'].shape[0]

scale = ((raw_data[0]*raw_data[1]).mean('stim').sum('unit')**.5).values
scale = 5#did this by hand for visualization
n_neur = raw_data.coords['unit'].shape[0]
# load sig_eigs
params = fit_df.loc[nm, ['fit_cvpca_w_log_c1', 'fit_cvpca_w_alpha1', ]]
A = list(params[['fit_cvpca_w_alpha1',]])
B = []
log_c1 = params['fit_cvpca_w_log_c1']
ind = np.arange(1, 1 + n_neur).astype(int)
cvpca_pl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))

params = fit_df.loc[nm, ['pl_b0_raw_log_c1', 'pl_b0_raw_alpha1', ]]
A = list(params[['pl_b0_raw_alpha1',]])
B = []
log_c1 = params['pl_b0_raw_log_c1']
ind = np.arange(1, 1 + n_neur).astype(int)
meme_pl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))

params = fit_df.loc[nm, ['pl_b1_raw_log_c1', 'pl_b1_raw_alpha1', 'pl_b1_raw_b1', 'pl_b1_raw_alpha2']]
A = list(params[['pl_b1_raw_alpha1', 'pl_b1_raw_alpha2']])
B = [int(params['pl_b1_raw_b1']),]
log_c1 = params['pl_b1_raw_log_c1']
ind = np.arange(1, 1 + n_neur).astype(int)
meme_bpl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))
sig_eigs = [cvpca_pl*scale**2, meme_pl*scale**2, meme_bpl*scale**2]

#load eigenmoments
mom_dist = xr.open_dataarray('./mom_dist.nc')
mom_est = xr.open_dataarray('./mom_est.nc')

k_moms = 10
eig_mom = mom_est.loc[nm][:k_moms]
eig_mom_bs = mom_dist.loc[nm][..., :k_moms]
eig_ind = np.arange(1, k_moms+1)

eig_mom = rescale_eig_mom(eig_mom, scale**(-1))
eig_mom_bs_resc = np.array([rescale_eig_mom(eig_mom_bs[i], scale**(-1)) for i in range(eig_mom_bs.shape[0])])

#%% FIG 3E-H
eig_nms = ['cvPCA power law', 'MEME power law', 'MEME broken power law']
#plot sig_eigs next to each other
for legends in [True, False]:
    plt.figure(figsize=(8,2.6))
    plt.subplot(141)
    plt.title('Example recording\nfit eigenspectra')
    colors = ['blue', 'pink', 'red']
    alphas = [1, 1., 1]
    for i, eig_spec in enumerate(sig_eigs):
        plt.loglog(ind, eig_spec, label=eig_nms[i], color=colors[i], alpha=alphas[i])
        #plt.loglog(ind, eig_spec, label=eig_nms[i], color=colors[i])
    if legends:
        plt.legend(loc=(0,2), framealpha=1)
    plt.ylabel(r'$\lambda_i$')
    plt.xlabel(r'i (rank)')
    plt.axis('square')

    plt.xticks([10**float(i) for i in range(-1, 6, 2)])
    plt.yticks([10**float(i) for i in range(-4, 2, 2)])
    plt.xlim(1e-1, 1e5)
    plt.ylim(1e-5, 10)
    plt.grid()
    #small annotation in bottom left of fn
    #plt.annotate(nm, xy=(0.01, 0.01), xycoords='axes fraction', fontsize=3)

    plt.subplot(142)
    #set face color to none
    plt.errorbar(eig_ind, eig_mom, yerr=eig_mom_bs_resc.std(0)*1.94, c='grey', marker='none', mfc='none', zorder=10);plt.semilogy()
    if legends:
        plt.legend(['Unbiased estimate ' r'$\pm$' ' 95% CI'], loc=(1,1.5), framealpha=1)
    for i, eig_spec in enumerate(sig_eigs):
        fit_eig_mom = [(eig_spec**p).mean() for p in range(1, k_moms+1)]
        plt.plot(eig_ind, fit_eig_mom, c=colors[i], marker='.', alpha=alphas[i]);plt.semilogy()

    plt.ylabel('Eigenmoment')
    plt.xlabel('Eigenmoment power')
    plt.title('Model vs data \n eigenmoments')
    plt.ylim(1e-5, 1e-2)
    plt.xlim(0, 11)

    sig_eigs_bpl = []
    for i in range(len(fit_df)):
        nm = fit_df.iloc[i]['file_name']
        scale = 5
        raw_data = xr.open_dataset(resp_data_dir + nm + '.nc')['resp']
        n_neur = raw_data.coords['unit'].shape[0]

        params = fit_df.loc[nm].loc[['pl_b1_raw_log_c1', 'pl_b1_raw_alpha1', 'pl_b1_raw_b1', 'pl_b1_raw_alpha2']]
        A = list(params[['pl_b1_raw_alpha1', 'pl_b1_raw_alpha2']])
        B = [int(params['pl_b1_raw_b1']),]
        log_c1 = params['pl_b1_raw_log_c1']
        ind = np.arange(1, 1 + n_neur).astype(int)
        meme_bpl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))

        sig_eigs_bpl.append(meme_bpl*scale**2)
    plt.xticks(range(1, k_moms+1, 2))
    plt.subplot(143)
    for i, eig_spec in enumerate(sig_eigs_bpl):
        ind = np.arange(1, len(eig_spec)+1)
        plt.loglog(ind, eig_spec, label=fit_df.iloc[i]['file_name'].split('ms_')[1], c='r', lw=0.25)
        #plt.loglog(ind, eig_spec, label=eig_nms[i], color=colors[i])

    plt.annotate(r'$\alpha_1$', xy=(3, 1), fontsize=12, xycoords='data')
    plt.annotate(r'$\alpha_2$', xy=(500, 0.01), fontsize=12, xycoords='data')
    #plt.legend(fontsize=4, loc=(0.5,0.6))
    plt.ylabel(r'$\lambda_i$')
    plt.xlabel(r'i (rank)')
    plt.axis('square')
    plt.xlim(1e-1, 1e5)
    plt.ylim(1e-5, 10)
    plt.xticks([10**float(i) for i in range(-1, 6, 2)])
    plt.yticks([10**float(i) for i in range(-4, 2, 2)])
    plt.title('Broken power law\nacross recordings')
    plt.grid()

    plt.subplot(144)
    plt.errorbar(fit_df['fit_cvpca_w_alpha1'], fit_df['pl_b1_raw_alpha2'], 
                yerr=ci_df['pl_b1_raw_alpha2'].groupby('fn_nms').std()*1.94, linestyle='none', marker='.', c='k',
                ecolor='k', elinewidth=0.5, capsize=0.5, markeredgewidth=0.5, markeredgecolor='w', zorder=10)

    ll = 0.9
    ul =1.3
    plt.axis('square')
    plt.ylim(ll,ul);plt.xlim(ll, ul)
    ticks = np.linspace(ll, ul, 5)
    ticks = [0.9, 1, 1.1, 1.2, 1.3]
    plt.yticks(ticks)
    plt.xticks(ticks)
    plt.plot([ll,ul], [ll,ul], c='k')
    plt.grid()
    plt.xlabel('cvPCA ' r'$\alpha$')
    plt.ylabel('MEME ' r'$\alpha_2$')
    plt.title('Comparison of\ntail slope')
    plt.tight_layout()

    plt.savefig('./fit_str_data_legend=' + str(legends) + '.pdf', 
                                    bbox_inches='tight', transparent=True)
#%% SI for X^2 test
# first get parameters for each recording: cvPCA, MEME, and MEME with broken power law
colors = ['blue', 'pink', 'red']
#fit_df = fit_df.set_index('fn_nms')
#load the bootstrapped eigenmomemnt distributions
mom_dist = xr.open_dataarray( './mom_dist.nc')
mom_est = xr.open_dataarray( './mom_est.nc')
k_moms = 10
stats = []
for nm in fit_df['file_name']:
    #load the raw data
    raw_data = xr.open_dataset(resp_data_dir + nm + '.nc')['resp']
    n_neur = raw_data.coords['unit'].shape[0]



    #get original ests and  whitening matrix from mom_dist for 10 moments
    est_eig = mom_est.loc[nm][..., :k_moms].values
    bs_est_eig = mom_dist.loc[nm][..., :k_moms].values
    W = em.bs_eig_mom_cov_W(transform='raw', bs_est_eig=bs_est_eig, return_W=True)

    #get cvPCA as power law
    A = [fit_df.loc[nm, 'fit_cvpca_w_alpha1'],]
    B = []
    log_c1 = fit_df.loc[nm, 'fit_cvpca_w_log_c1']
    ind = np.arange(1, 1 + n_neur).astype(int)
    cvpca_pl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))
    #get MEME as power law
    A = [fit_df.loc[nm, 'pl_b0_raw_alpha1'],]
    B = []
    log_c1 = fit_df.loc[nm, 'pl_b0_raw_log_c1']
    meme_pl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))
    #get MEME with broken power law
    A = [fit_df.loc[nm, 'pl_b1_raw_alpha1'], fit_df.loc[nm, 'pl_b1_raw_alpha2']]
    B = [int(fit_df.loc[nm, 'pl_b1_raw_b1'])]
    log_c1 = [fit_df.loc[nm, 'pl_b1_raw_log_c1'],]
    meme_bpl = np.exp(em.get_bpl_func_all(A, log_c1, B, ind))

    dof = [2, 2, 4]
    pls = [cvpca_pl, meme_pl, meme_bpl]
    eig_mom = np.array([[np.mean(pl**(p+1)) for p in range(k_moms)] for pl in pls])
    _ = []
    for i, pl in enumerate(pls):
        a_eig_mom = eig_mom[i]
        chi2_stat = ((W @ (est_eig - a_eig_mom)[:,None])**2).sum()
        #look up the p-value of chi2 statistic using dof
        p = 1 - chi2.cdf(chi2_stat, k_moms-dof[i])
        _.append([p, chi2_stat])
    stats.append(_)
stats = np.array(stats)
print(stats.shape)
#plot the p-values as a trace for each recording

plt.figure(figsize=(2,2))
plt.plot(stats[:,0,1], label='cvPCA', color= colors[0],lw=1)
plt.plot(stats[:,1,1], label='MEME', color= colors[1], lw=1)
plt.plot(stats[:,2,1], label='MEME with broken power law', color= colors[2], lw=1)
plt.semilogy()
#plt.legend()
#xlabel is the chi squared symbol
plt.xlabel('Recording')
plt.ylabel(r'$\chi^2$')
#make ylim between 1.1 * max and 0.01
plt.ylim(0.1, 10 * stats.max())
#now make scatter plots on the same traces with open dots for where p > 0.01
for i in range(stats.shape[0]):
    for j in range(stats.shape[1]):
        if stats[i,j,0] > 0.001:
            plt.scatter(i, stats[i,j,1], marker='x', color='k', s=10, zorder=10)
        else:
            plt.scatter(i, stats[i,j,1], marker='.', color='k', s=5, zorder=10)
plt.xticks(np.arange(0, stats.shape[0], 1))
#make the xtick labels follow the convention: ms_natimg2800_M160825_MP027_2016-12-14 - > MP030_2017-05-29
labels = fit_df['file_name'].values
labels = [l.split('_')[-2] + '_' + l.split('_')[-1] for l in labels]
plt.gca().set_xticklabels(labels, rotation=90, fontsize=6)
fn_nms = fit_df.index
fn_nms = [(f.split('_')[-1]) for f in fn_nms]
#plt.gca().set_xticklabels(fn_nms, rotation=90, fontsize=6)
plt.yticks([1, 1e6, 1e12, 1e18])
#create custom legend where black dots are sig dif and x's are not sig dif
plt.legend(handles=[plt.Line2D([0], [0], marker='.', color='k', lw=0, markersize=5, label=r'$p \leq 0.001$'),
                    plt.Line2D([0], [0], marker='x', color='k', lw=0, markersize=5, label=r'$p > 0.001$'),], 
                    loc=(1.01,0.), framealpha=1, title=r'$\chi^2$' + ' test')
plt.savefig('./chi_squared_eig_mom_fits.pdf', 
                                bbox_inches='tight', transparent=True)

# %%
prs_bpl = []
prs_pl1 = []
for i, eig_spec in enumerate(sig_eigs_bpl):
    #get participation ratio of eig_spec
    ind = np.arange(1, len(eig_spec)+1).astype(float)
    pr = (eig_spec).sum()**2/(eig_spec**2).sum()
    prs_bpl.append(pr)
    pl_eig_spec = ind**(-1.)
    pr = (pl_eig_spec).sum()**2/(pl_eig_spec**2).sum()
    prs_pl1.append(pr)

#plt.loglog(ind, eig_spec, label=eig_nms[i], color=colors[i])
# %%
plt.plot(prs_bpl, 'r', label='broken power law')
plt.plot(prs_pl1, 'b', label='power law')
plt.ylim(0,None)
plt.legend()

#%%
print('Average ratio of PR BPL to PL=1:', np.round((np.array(prs_bpl)/np.array(prs_pl1)).mean(), 2))
# %%
for i, eig_spec in enumerate(sig_eigs_bpl):
    ind = np.arange(1, len(eig_spec)+1)
    eig_spec = eig_spec/eig_spec.sum()
    plt.loglog(ind, eig_spec, label=fit_df.iloc[i]['file_name'].split('ms_')[1], c='r', lw=0.25)
    eig_spec = ind**-1.
    eig_spec = eig_spec/eig_spec.sum()
    plt.loglog(ind, eig_spec, label=fit_df.iloc[i]['file_name'].split('ms_')[1], c='k', lw=0.25)
#%%
#get average number of neurons for each recording
n_neurs = []
for nm in fit_df['file_name']:
    #load the raw data
    raw_data = xr.open_dataset(resp_data_dir + nm + '.nc')['resp']
    n_neur = raw_data.coords['unit'].shape[0]
    n_neurs.append(n_neur)
print('Average number of neurons:', np.round(np.array(n_neurs).mean(), 2))
np.max(n_neurs)
#%%
plt.figure(figsize=(4,3))
frac_var = 0.75
ind_frac_vars_bpl = []
ind_frac_vars_pl = []
for i, eig_spec in enumerate(sig_eigs_bpl):
    ind = np.arange(1, len(eig_spec)+1)
    eig_spec = eig_spec/eig_spec.sum()
    plt.plot(ind, eig_spec.cumsum(), label='Fit broken power laws', c='r',)
    ind_frac_var = ind[eig_spec.cumsum() < frac_var][-1]
    ind_frac_vars_bpl.append(ind_frac_var)
    plt.grid()
    plt.semilogx()
    
    eig_spec = ind**-1.
    eig_spec = eig_spec/eig_spec.sum()
    #plot a vertical line where the variance explained is 0.8
    ind_frac_var_pl = ind[eig_spec.cumsum() < frac_var][-1]
    ind_frac_vars_pl.append(ind_frac_var_pl)
    #plt.plot([ind_frac_var_pl, ind_frac_var_pl], [0, frac_var], c='k')
    if i ==0:    
        #plot for example visualizations without too much crowding.
        
        ind = np.arange(1, 10000)
        eig_spec = ind**-1.
        eig_spec = eig_spec/eig_spec.sum()
        plt.plot(ind, eig_spec.cumsum(), label='Power law slope=1', c='k',)
        plt.legend()

    plt.xlabel('Eigenvector rank')
    plt.ylabel('Cumulative fraction variance explained')
plt.yticks([0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], )
plt.xticks([1, 10, 100, 1000, 10000], ['1', '10', '100', '1,000', '10,000'], )
plt.savefig('./fig_S2.pdf',
                                bbox_inches='tight', transparent=True)
# %%
plt.plot(ind_frac_vars_bpl, 'o')
plt.plot(ind_frac_vars_pl, 'o')
plt.ylabel('i where cumulative frac var = 0.8')
plt.xlabel('Recording')

plt.ylim(0, None)
#get the average ratio of ind_frac_var_pl to ind_frac_var
print('Average ratio of PL to BPL:', np.round((np.array(ind_frac_vars)/np.array(ind_frac_var_pl)).mean(), 2))
# %%
#average number for bpl
print('Average number of eigenvectors for BPL:', np.round(np.array(ind_frac_vars_bpl).mean(), 2))
print('Average number of eigenvectors for PL:', np.round(np.array(ind_frac_vars_pl).mean(), 2))
# %%

