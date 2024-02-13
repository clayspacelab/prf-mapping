import os as os
import ctypes
import pickle
import multiprocessing as mp

import tqdm
import numpy as np
from scipy.io import loadmat
from scipy.signal import detrend
import nibabel as nib

import popeye.utilities as utils
from popeye import css
from popeye.visual_stimulus import VisualStimulus, resample_stimulus

try:
	mp.set_start_method('fork')
except:
	pass

# some params
native_width = 36.3 # assuming this from prisma measurements i made at NYU
viewing_distance = 64.53 # assuming this from prisma measurements i made at NYU
tr_length = 1.3 # 1300 ms per file name
dtype = ctypes.c_int16 # for stimulus
scale_factor = 1 # for speed
resample_factor = 0.84 # for speed

# load stimulus
bar = loadmat('Stimuli/bar_stimulus_masks_1300ms_images.mat')['images']
params = loadmat('Stimuli/bar_stimulus_masks_1300ms_params.mat')

# create stimulus object
bar = resample_stimulus(bar, resample_factor)
stimulus = VisualStimulus(bar, viewing_distance, native_width, scale_factor, tr_length, dtype)

# load BOLD data
# nii = nib.load('bar_seq_1_surf_25mm_mean_detrend.nii.gz')
nii = nib.load('sample_data/JC/RF1/JC_RF1_vista/bar_seq_1_ss5.nii.gz')
dat = nii.get_fdata()
# dat = np.asanyarray(nii.dataobj)

# create a mask using maximum intensity projection approach
# mask = nib.load('cerebralcortex_mask_resampled_25mm.nii.gz').get_data() != 0
# mask = nib.load('roi_hhl/bilat.V1V2V3.nii.gz').get_data() != 0
# mask = nib.load('/Users/aditya/Documents/GitHub/prf-mapping/sample_data/JC/surfanat_brainmask_master.nii.gz').get_data() != 0
mask = nib.load('sample_data/JC/surfanat_brainmask_master.nii.gz').get_fdata() != 0
[xi,yi,zi] = np.nonzero(mask)
indices = [(xi[i],yi[i],zi[i]) for i in range(len(xi))]

print( 'length')
print( len(xi) )

print('getenv')
print(int(os.getenv('SLURM_CPUS_PER_TASK')))

# extract timeseries of interest
# linear detrend
# add the pre-detrend mean back in (otherwise you can't accurately calculate % signal change)
# convert to % signal change
# bold = dat[mask]
# bold_mu = np.mean(bold,-1)
# bold_dt = detrend(bold)
# bold_psc = utils.percent_change(bold_dt+bold_mu[...,np.newaxis])

bold_psc = dat[mask]

# cleanup the volume data (optional)
del dat

# create model
model = css.CompressiveSpatialSummationModel(stimulus, utils.spm_hrf)

# set one hrf delay across the analysis
model.hrf_delay = 0

model.mask_size = 1

# set search grid
x_grid = (-12,12)
y_grid = (-12,12)
s_grid = (1/stimulus.ppd, 12)
n_grid = (0.1, 5)
grids = (x_grid, y_grid, s_grid, n_grid,)    

# set search grid
x_bounds = (-30,30)
y_bounds = (-30,30)
sigma_bounds = (1/model.stimulus.ppd, 15)
n_bounds = (1e-1, 5)
beta_bounds=(1e-8, None)
baseline_bounds = (None,None)
bounds = (x_bounds, y_bounds, sigma_bounds, n_bounds, beta_bounds, baseline_bounds,)

# fit settings
Ns = 10
auto_fit = 1
verbose = 0
print(bold_psc.shape)
print(bold_psc)
print(Ns^len(grids))

# gather
bundle = utils.multiprocess_bundle(css.CompressiveSpatialSummationFit, model, bold_psc, grids, bounds, indices, auto_fit, verbose, Ns)

# fit
ncpu = int(os.getenv('SLURM_CPUS_PER_TASK'))
with mp.Pool(ncpu) as pool:
	results = list(tqdm.tqdm(pool.imap(utils.parallel_fit, bundle), total=len(bundle)))

# save results
f = open('results/css_results.pkl', 'wb')
pickle.dump(results, f)

# example of how to load the results
f = open('results/css_results.pkl', 'rb')
results = pickle.load(f)

# screen out failed fits
output = [r for r in results if r is not None]

# save polar as nifti
nifti = utils.recast_estimation_results(output, nii, overloaded=True)
# nib.save(nifti, 'prf_css_polar_cerebralCortex_25mm_mean_detrend_8.nii.gz')
nib.save(nifti, 'results/prf_css.nii.gz')
