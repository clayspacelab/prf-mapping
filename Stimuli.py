import ctypes
import numpy as np
from scipy.io import loadmat
from popeye.visual_stimulus import VisualStimulus, resample_stimulus


# some params
native_width = 36.3 # assuming this from prisma measurements i made at NYU
viewing_distance = 64.53 # assuming this from prisma measurements i made at NYU
tr_length = 1.3 # 1300 ms per file name
dtype = ctypes.c_int16 # for stimulus
scale_factor = 1 # for speed
resample_factor = 0.35 # for speed

# load stimulus
bar = loadmat('Stimuli/bar_stimulus_masks_1300ms_images.mat')['images']
params = loadmat('Stimuli/bar_stimulus_masks_1300ms_params.mat')

bar = resample_stimulus(bar, resample_factor)
stimulus = VisualStimulus(bar, viewing_distance, native_width, scale_factor, tr_length, dtype)