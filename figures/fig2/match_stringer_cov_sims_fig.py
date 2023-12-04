
#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
sys.path.append('../../src/')
import eig_mom as em
sim_dir = './meme_vs_cvpca_pl_est_match_str_sim/'
fit_df = pd.read_csv('../fig3/str_pt_estimates.csv')

#concatenate all the sims into a dataframe with their rec number 
sims = [sim for sim in os.listdir(sim_dir) 
                if 'rec' in sim and 'bsrun_1' in sim]
recs = []
sim_dfs = []
for sim in sims:
    rec = int(sim.split('_')[1])
    recs.append(rec)
    sim_dfs.append(pd.read_csv(sim_dir + sim, index_col=0,))
#turn last row into column for each dataframe this is temporary because of a saving error
nms = []
for i, sim_df in enumerate(sim_dfs):
    nms.append(sim_df.iloc[-1][0])
    sim_df = sim_df.iloc[:-1]
    sim_df.iloc[:, :-1] = sim_df.iloc[:, :-1].astype(float)
    sim_dfs[i] = sim_df

# concatentae sim_dfs into a single dataframe with rec as the outer index
# and the inner index being the simulation number
sim_df = pd.concat(sim_dfs, keys=nms, names=['fn_nms', 'sim'])

cov_types = ['aligned', 'orig', 'ind']
cov_types = ['aligned', 'ind']
truth = fit_df.set_index('fn_nms').loc[:, 'fit_cvpca_w_alpha1']
est_types = ['fit_cvpca_w_alpha1', 'pl_b0_raw_alpha1']
labels = ['cvPCA', 'MEME']

colors = ['b', 'r']
#now make errorbar scatter of simulations vs truth for cvpca and meme
fig, ax = plt.subplots(1, 2, figsize=(4, 2), )
for j, cov_type in enumerate(cov_types):
    
    for i, est_type in enumerate(est_types):
        nm = est_type + '_SN_cov_' + cov_type
        y = sim_df.loc[:, nm].groupby('fn_nms').mean()
        yerr = sim_df.loc[:, nm].groupby('fn_nms').std()
        truth = truth.loc[y.index]
        ax[j].errorbar(x = truth, y=y, 
                        yerr=yerr, fmt='.', label=labels[i], color=colors[i])
# make axis equal from 0 to 1.1 on both axes
#convert below into ax format
for j, cov_type in enumerate(cov_types):
    ax[j].plot([0.5, 1.25], [0.5, 1.25], 'k')
    ax[j].set_xlim([0.5, 1.25])
    ax[j].set_ylim([0.5, 1.25])
    ax[j].set_xticks([0.5, 0.75, 1, 1.25])
    ax[j].set_yticks([0.5, 0.75, 1, 1.25])

    ax[j].axis('square')
    ax[j].grid()

for j in range(1, 2):
    ax[j].set_yticklabels([])
    ax[j].set_xticklabels([])

ax[0].legend()
ax[0].set_xlabel('True ' r'$\alpha$')
ax[0].set_ylabel('Estimated ' r'$\alpha$')
ax[0].set_title('Signal & noise' + '\neigenvectors\naligned')
ax[1].set_title('Signal & noise' + '\nindependently sampled')
plt.savefig('./sim_match_str_meme_vs_cvpca.pdf', 
                                bbox_inches='tight', transparent=True)
# %%
