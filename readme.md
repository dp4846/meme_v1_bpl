# Code: The geometry of representation in mouse visual cortex obeys a broken power law

## Overall organization of code
Top level directory is: ./meme_v1_bpl/. The three directories below it are:

- `src/` 
    - contains the main function used to analyse and simulate, including the cvPCA estimator and MEME estimator. 

- `data/` 
    - contains the scripts to generate files that one might keep on a separate drive or download directly instead of generating themselves. Within this folder is a text file 'data_dir' that has the full path to the directory, '/stringer_2019/' which contains all large data files including the original data from the Stringer 2019.

- `figures/` 
    - Has a folder for each figure in the paper (5) and within each are scripts that either generate figures or data. The data scripts have '_data' appended to them and take a non-neglibible amount of time to run and then save processed data or simulation results directly into the figure folder.  The figure files have '_fig' appended to them and they read in this processed data to generate figures for the paper. The comments in these figure scripts indicate the panel being saved (e.g., Fig 2A). 



## Detailed description of how to reproduce results from paper
### Processing raw data
1. Download original data from:
[Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images](https://figshare.com/articles/dataset/Recordings_of_ten_thousand_neurons_in_visual_cortex_in_response_to_2_800_natural_images/6845348)
to a folder: `'./stringer_2019/orig_stringer_2019_data/'` then update
`'./data/data_dir.txt'` to the directory that `'/stringer_2019/'` is in.

2. Run
`'./data/convert_stringer_mat_to_xr.py'`
to convert from the .mat format to xarray dataset format.
    -  Files are saved to: `'./stringer_2019/processed_data/neural_responses/'` prepended with:
        -   raw_ is just the raw responses
        -   ms_ is the mean subtracted responses
        -   spont_sub_ is the mean subtracted responses with spontaneous activity subtracted (see Stringer 2019)
        
        All recordings including cell position (unsure of exact units).
For recordings that have it, the tdtomato+ labels are also saved in a csv file in
`'./stringer_2019/processed_data/red_cell/'`

### Estimating signal and noise covariance matrices
For simulations we need estimates of signal and noise covariance from the original data, run: `'./data/est_mean_SN_cov_stringer.py'` saves to: `'./processed_data/stringer_sn_covs/'`.

### Figures

#### Figure 1: 
Conceptual figure describing issues involved in estimating eigenspectra. No data files needed.

#### Figure 3:
Estimates of power-laws from neural data (this has to be run before Figure 2 scripts because those simulations are based on estimates of the eigenspectra of the neural data).
1. `'figures/fig3/est_sig_eig_spec_data.py'`
first calculates eigenmoments and bootstrap samples of eigenmoments
saved in folder respectively as 
`'./mom_est.nc'` and `'./mom_dist.nc'`
then estimates power laws using cvPCA and power laws and broken powerlaws using MEME. Saves these in csv
`'str_pt_estimates_all.csv'`
2. Then for confidence intervals we use 
`'bpl_ci_data.py'`
that saves to:
`'bpl2_sims'`
then generates `'est_sig_eig_spec_figs.py'`

- [ ] where does bpl2_sims come from? these are used as CI's in Fig 3H
I found it in parameteric_bootstrap_bpl2.py in the original eigenspectra file

#### Figure 2:
Multivariate normal simulations from true power-laws.
1. `'small_power_law_sims_'` run small simulations (~100 neurons) to demonstrate differences between MEME and cvPCA.
2. `'match_stringer_cov_sims_data'` run simulations matching signal and noise covariance from original data saves files to ???.

where does `rec_0_ms_natimg2800_M160825_MP027_2016-12-14_bsrun_1_pbci_pl.csv`
come from? these are used in Fig 2 EF
it is in the folder 
`match_stringer_cov_sims_data`
I switched it to: `"{sim_dir}rec_{rec}_{nm}_pbci_pl.csv`

#### Figure 4:

1. `'eigenmode_tuning_data.py'` generates the data that is the basis of all Figure 4 and the neural data results in Figure 5. Because these files are large they are saved to: 
`'./stringer_2019/processed_data/eig_tuning/'`

-   [ ] why does eig_tuning_dir + 'neur_snr_' have two rows?
-   [ ] why does eig_tuning_dir + 'pc_snr_'  have two columns?



#### Figure 5:
no _data files, uses results from Figure 4: `'eigenmode_tuning_data.py'` saved to `'./stringer_2019/processed_data/eig_tuning/'`



### GENERAL TODO

I am currently just making sure all figures can be made and noting where data should be coming from and putting it all into _data files. text with a check box is a todo item.

- [X] replace figure 5 with new figure.
- [ ] supplementary figure  of population SNR differences.
- [ ] supplementary figure of gabor regression for neurons and PCs.
- [ ] choose one normalization method, scale by var for example.
- [ ] put src test into v1_bpl_sims.
- [ ] get recording X corresponding to actual id.
- [ ] use raw neural data recordings.
- [ ] use correct mean subtraction method.
- [ ] use participation ratio.
