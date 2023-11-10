import dask.array as da
import dask
import pickle
from dask.distributed import Client
from scipy.io import loadmat
from scipy.signal import detrend
import numpy as np
import popeye.utilities as utils
from popeye import css
from popeye.visual_stimulus import VisualStimulus, resample_stimulus

# Loading Stimulus Data
bar = loadmat('Stimuli/bar_stimulus_masks_1300ms_images.mat')['images']
params = loadmat('Stimuli/bar_stimulus_masks_1300ms_params.mat')
bar = resample_stimulus(bar, resample_factor)
stimulus = VisualStimulus(bar, viewing_distance, native_width, scale_factor, tr_length, dtype)

# Loading BOLD Data
nii = nib.load('/path/to/bold_data.nii.gz')
dat = nii.get_fdata()
mask = nib.load('/path/to/mask.nii.gz').get_fdata() != 0
[xi, yi, zi] = np.nonzero(mask)
indices = [(xi[i], yi[i], zi[i]) for i in range(len(xi))]

# Defining Parameters
x_grid = (-12, 12)
y_grid = (-12, 12)
s_grid = (1 / stimulus.ppd, 12)
n_grid = (0.1, 5)
grids = (x_grid, y_grid, s_grid, n_grid)

x_bounds = (-30, 30)
y_bounds = (-30, 30)
sigma_bounds = (1 / model.stimulus.ppd, 15)
n_bounds = (0.1, 5)
beta_bounds = (1e-8, None)
baseline_bounds = (None, None)
bounds = (x_bounds, y_bounds, sigma_bounds, n_bounds, beta_bounds, baseline_bounds)


# Convert numpy array to Dask array
bold_psc_dask = da.from_array(bold_psc, chunks='auto')  # Adjust the chunk size as needed

# Create Dask delayed objects for processing chunks
@dask.delayed
def process_chunk(idx):
    bold_psc_chunk = bold_psc_dask[idx]
    return utils.parallel_fit(css.CompressiveSpatialSummationFit, model, bold_psc_chunk, grids, bounds, [idx], auto_fit, verbose)

# Process chunks using Dask parallelization
results = []
for idx in range(len(indices)):
    result = process_chunk(idx)
    results.append(result)

# Compute the Dask delayed objects to get the results
results = dask.compute(*results)

# Flatten the results list
output = [fit for sublist in results for fit in sublist]

# Save results to a pickle file
with open('css_results.pkl', 'wb') as f:
    pickle.dump(output, f)

# ... (Rest of your code)
