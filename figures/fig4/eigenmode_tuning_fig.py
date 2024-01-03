#%%
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io as sio
import xarray as xr
from tqdm import tqdm

data_dir = '/Volumes/dean_data/neural_data/stringer_2019/'
orig_data_dir = data_dir + 'orig_stringer2019_data/'
resp_data_dir = data_dir + 'processed_data/neural_responses/'
eig_tuning_dir = data_dir + 'processed_data/eig_tuning/'
rf_coords = {'ms_natimg2800_M160825_MP027_2016-12-14':[(25, 65), (35, 85)],
             'ms_natimg2800_M161025_MP030_2017-05-29':[(25, 65), (95, 145)],
            'ms_natimg2800_M170604_MP031_2017-06-28':[(25, 65), (90, 135)],
            'ms_natimg2800_M170714_MP032_2017-08-07':[(20, 55), (20, 60)],
            'ms_natimg2800_M170714_MP032_2017-09-14':[(10, 45), (90, 130)],
            'ms_natimg2800_M170717_MP033_2017-08-20':[(5, 50), (90, 150)],
            'ms_natimg2800_M170717_MP034_2017-09-11':[(20, 60), (90, 135)]}
#%%

fns = [data_dir + 'xr_conv/' + fn for fn in os.listdir(resp_data_dir) if 'natimg2800_M' in fn and not 'npy' in fn and 'ms' in fn]
fn = 'ms_natimg2800_M170717_MP033_2017-08-20'
(r1, r2), (c1, c2) = rf_coords[fn]
#fn = fn.split('/')[-1].split('.')[0]
v_r_neur = np.load(eig_tuning_dir + 'sp_cov_neur_u_r_' + fn + '.npy', )# eig vecs neurons
v_r_stim = np.load(eig_tuning_dir + 'sp_cov_stim_u_r_' + fn + '.npy', )# eig vecs stim
v_r_neur[:, 0] = -v_r_neur[:, 0]#flip the first eigenvectors to be positive (arbitrary)
#v_r_stim[:, 0] = -v_r_stim[:, 0]
snr_neur = np.load(eig_tuning_dir + 'neur_snr_' + fn + '.npy')[0]
#rf = np.load(eig_tuning_dir + 'rf_est_' + fn + '.npy', )
rf = np.load(eig_tuning_dir + 'eig_rf_est_' + fn + '.npy', )
rf[0] = -rf[0]#similarly flip the first rf
#l_r = np.load(eig_tuning_dir + 'lin_rf_resp_' + fn + '.npy',)#this is now eig_lin_resp
l_r = np.load(eig_tuning_dir + 'eig_lin_resp_' + fn + '.npy',)#this is now eig_lin_resp
#l_r[:, 0] = -l_r[:, 0]
r2_pcs = np.load(eig_tuning_dir + 'lin_r2_pc_' + fn + '.npy')

imgs = sio.loadmat(orig_data_dir+ 'images_natimg2800_all.mat')['imgs']
img_order = xr.open_dataset(resp_data_dir + fn + '.nc')['resp'].coords['stim'].values
imgs = imgs[..., img_order]

#%%
#get percent negative loadings of first eigenvector
print('Percent negative loadings of first eigenvector: ' + str(np.round(np.sum(v_r_neur[:,0]>0)/len(v_r_neur[:,0])*100, 2)) + '%')
#%% fig 4a-b
n_ims = 100
#eigenvectors, eigenmode tuning and linear component (5a-b)
nrows = 4

sc = 1.5
fig = plt.figure(constrained_layout=False, figsize=(8*sc, 0.8*sc*nrows), dpi=500)
gs0 = fig.add_gridspec(nrows, 6, hspace=0.1)
lin_color = 'orange'

coef_gs = [gs0[i, :3].subgridspec(1, 2, wspace=0.03, hspace=0) for i in range(nrows)]  
coef_axs = [[fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 0])] for gs in coef_gs]
for row in range(nrows):
    coef_axs[row][1].plot(np.arange(0, len(snr_neur)), v_r_neur[:, row][np.argsort(snr_neur)[::-1]], 
                    color='k', lw=0.5)
    coef_axs[row][0].plot(v_r_stim[:n_ims, row], c='k', label='Eigenmode tuning')
    coef_axs[row][0].plot(l_r[:n_ims, row], c=lin_color, label='Linear component')
    coef_axs[row][0].annotate(r'$R^2$ = ' + str(np.round(r2_pcs[row], 2)), (0.95, 0.95), xycoords='axes fraction', fontsize=8, ha='right', va='top')
    for i in range(2):
        lim = np.max(np.abs(coef_axs[row][i].get_ylim()))
        coef_axs[row][i].set_ylim(-lim, lim)
        coef_axs[row][i].set_yticks([0,])
        coef_axs[row][i].grid(axis='y')
        coef_axs[row][0].set_yticklabels([])
        coef_axs[row][1].set_yticklabels(['0',])
    if row==0:
        coef_axs[row][0].legend(fontsize=8, loc=(0.2,0), frameon=False, 
                        labelspacing=0.01)
        coef_axs[row][0].set_title('Eigenmode tuning')
        coef_axs[row][1].set_title('Eigenmode neural loading')
    
    if row==(nrows-1):
        coef_axs[row][1].set_xlabel('SNR of neuron ')
        coef_axs[row][1].set_xlabel('Neuron (SNR ranked)')
        coef_axs[row][0].set_xlabel('Stimulus')
    else:
        for i in range(2):
            coef_axs[row][i].set_xticklabels([])

#fig 4c - linear RF
rf_axs = [fig.add_subplot(gs0[i, 3]) for i in range(nrows)]
[ax.imshow(rf[r1:r2, c1:c2,i], cmap='Greys_r', 
           vmin=-np.max(np.abs(rf[r1:r2, c1:c2, i])), 
           vmax=np.max(np.abs(rf[r1:r2, c1:c2, i]))) 
           for i, ax in enumerate(rf_axs)]
[(ax.set_xticks([]), ax.set_yticks([]), ) for i, ax in enumerate(rf_axs)]
rf_axs[0].set_title('Linear RF')
for ax in rf_axs:
    for spine in ax.spines.values():
        spine.set_edgecolor(lin_color)
        spine.set_linewidth(3)

#fig 4d - response ranked images
img_gs = [ gs0[i, 4:].subgridspec(2, 6, wspace=0, hspace=0.05) for i in range(nrows)]
img_axs = np.zeros((nrows, 2, 6)).tolist()
for row, gs in enumerate(img_gs):
    for i in range(2):
        for j in range(6):
            ax = fig.add_subplot(gs[i, j])
            ax.set_xticks([])
            ax.set_yticks([])
            img_axs[row][i][j] = ax
    
    for q in [0, 1, 2, -3, -2, -1]:
        if row==0:
            if q==0:
                img_axs[row][1][q].set_ylabel('Lin.')
                img_axs[row][0][q].set_ylabel('Eig.')
                img_axs[row][0][q].set_title('                                                    Response ranked\n1' )

            else:
                if (q+1)>0:
                    img_axs[row][0][q].set_title(str(q+1 ))
                elif (q+1)<0:
                    img_axs[row][0][q].set_title('M' + str(q+1))
                else:
                    img_axs[row][0][q].set_title('M')
        img_axs[row][0][q].imshow(imgs[r1:r2, c1:c2, v_r_stim[:, row].argsort()[::-1][q]], cmap='gray')
        img_axs[row][1][q].imshow(imgs[r1:r2, c1:c2,  l_r[:, row].argsort()[::-1][q]], cmap='gray')
        for spine in img_axs[row][1][q].spines.values():
            spine.set_edgecolor(lin_color)
            spine.set_linewidth(1)
plt.savefig('PCA_example'+fn+'.pdf', 
                                bbox_inches='tight', transparent=True, 
                                dpi=2000)

#%% fig 4F other example population linear receptive fields
fn_labs = [_.split('/')[-1].split('.')[0] for _ in fns]

n_eigs = 30
fig, axs = plt.subplots(7, n_eigs, figsize=(8,2))
for i, _ in enumerate(fns):
    rf_nm = _.split('/')[-1].split('.')[0]
    
    #rf = np.load(eig_tuning_dir + 'rf_est_' + rf_nm + '.npy', )
    rf = np.load(eig_tuning_dir + 'eig_rf_est_' + rf_nm + '.npy', )
    (r1, r2), (c1, c2) = rf_coords[rf_nm]
    rf = rf[r1:r2, c1:c2]
    for ind in range(n_eigs):
        #put a very small annotatio nof the rf_nm below each row
        if ind ==0:
            axs[i][ind].annotate(rf_nm, (0, -0.2), xycoords='axes fraction', fontsize=2.5)
        a_rf = rf[..., ind]
        mxmn = np.max([-a_rf.min(), a_rf.max()])
        
        axs[i][ind].imshow(a_rf, vmin=-mxmn, vmax=mxmn, cmap='gray')
        axs[i][ind].set_xticks([])
        axs[i][ind].set_yticks([])
        axs[i][ind]
        #remove borders
        for spine in axs[i][ind].spines.values():
            spine.set_visible(False)
        # if ind==0:
        #     plt.ylabel()
        # # plt.title(r'$R^2=$' + str(r2s_pc[ind].round(2)))
plt.tight_layout(pad=0, w_pad=0, h_pad=0)
    # plt.annotate(fn.split('_')[3] + '_' + fn.split('_')[4], (0, -0.5), xycoords='axes fraction', fontsize=8)
plt.savefig('./linear_rfs_all.pdf', bbox_inches='tight', transparent=True, dpi=1000)
#%% fig 4e toy signal svd figure
plt.figure()
u = np.random.normal(size=(8,1))
v = np.random.normal(size=(5,1))
plt.imshow(u @ v.T, cmap='gray')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.xticks([]);plt.yticks([])

plt.figure()
plt.imshow(u, cmap='gray')
plt.xticks([]);plt.yticks([])

plt.figure()
plt.imshow(v.T, cmap='gray')
plt.xticks([]);plt.yticks([])

full = np.random.normal(size=(8,5))
plt.figure()
plt.imshow(full + u @ v.T, cmap='gray')
plt.gca().set_xticklabels([])
plt.gca().set_yticklabels([])
plt.xticks([]);plt.yticks([])
