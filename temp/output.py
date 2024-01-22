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

# nii = nib.load('/Users/aditya/Documents/GitHub/prf-mapping/sample_data/JC/RF1/JC_RF1_vista/bar_seq_1_surf.nii.gz')
nii = nib.load('/Users/aditya/Documents/GitHub/prf-mapping/sample_data/JC/RF1/JC_RF1_vista/bar_seq_1_ss5.nii.gz')
# example of how to load the results
f = open('css_results.pkl', 'rb')
results = pickle.load(f)
print(len(results))
# screen out failed fits
output = [r for r in results if r is not None]
print(len(output))

# save polar as nifti
nifti = utils.recast_estimation_results(output, nii, overloaded=True)
nib.save(nifti, 'prf_css.nii.gz')