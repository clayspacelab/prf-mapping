import ctypes, multiprocessing
import numpy as np
import sharedmem
from popeye import css
import popeye.utilities as utils
from popeye.visual_stimulus import VisualStimulus, simulate_bar_stimulus
from matplotlib import pyplot

# seed random number generator so we get the same answers ...
np.random.seed(2764932)

### STIMULUS
## create sweeping bar stimulus
thetas = np.array([-1,0,90,180,270,-1]) # in degrees, -1 is blank
bar = simulate_bar_stimulus(100, 100, 40, 20, thetas, 30, 30, 10)

## create an instance of the Stimulus class
stimulus = VisualStimulus(bar, 50, 25, 0.50, 1.0, ctypes.c_int16)

### MODEL
## initialize the gaussian model
model = css.CompressiveSpatialSummationModel(stimulus, utils.double_gamma_hrf)
model.hrf_delay = 0 #This is additional delay; Note that double_gamma_hrf already has 5 sec delay by default

## generate a random pRF estimate
x = -5.24
y = 2.58
sigma = 1.24
n = 1.2
beta = 0.55
baseline = -0.88

## create the time-series for the invented pRF estimate
data = model.generate_prediction(x, y, sigma, n, beta, baseline)

## add in some noise
data += np.random.uniform(-data.max()/10,data.max()/10,len(data))

### FIT
## define search grids
# these define min and max of the edge of the initial brute-force search.
x_grid = (-10, 10)
y_grid = (-10, 10)
s_grid = (0.25, 12)
n_grid = (.1, 4) #exponent

## define search bounds
# these define the boundaries of the final gradient-descent search.
x_bound = (-12.0, 12.0)
y_bound = (-12.0,12.0)
s_bound = (1/stimulus.ppd, 12.0) #smallest sigma is a pixel
b_bound = (1e-8,None)
u_bound = (None,None)
n_bound = (1e-1, 4.0)

## package the grids and bounds
grids = (x_grid, y_grid, s_grid, n_grid)
bounds = (x_bound, y_bound, s_bound, n_bound, b_bound, u_bound,)

## fit the response
# auto_fit = True fits the model on assignment
# verbose = 0 is silent
# verbose = 1 is a single print
# verbose = 2 is very verbose
fit = css.CompressiveSpatialSummationFit(model, data, grids, bounds, Ns=3,
                     voxel_index=(1,2,3), auto_fit=True,verbose=2)

print(fit)
print(type)