import time
from ccpi.framework import ImageData, ImageGeometry, AcquisitionGeometry, \
    AcquisitionData
from ccpi.optimisation.algs import FISTA, FBPD, CGLS, SIRT
from ccpi.optimisation.funcs import Norm2sq, Norm1, TV2D, IndicatorBox, ZeroFun

from operators import *
import sys

sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/CCPi-astra/Wrappers/Python/ccpi/astra')
from ops import AstraProjectorSimple


#from ccpi.astra.ops import AstraProjectorSimple
from ccpi.optimisation.ops import PowerMethodNonsquare
from skimage.util import random_noise

import numpy as np
import matplotlib.pyplot as plt

from my_changes import *

#%% # Create phantom

N = 100
ig = ImageGeometry(voxel_num_x=N,voxel_num_y=N)

x = ImageData(create_toy_phantom(N, ig))

detectors = N
angles = np.linspace(0,np.pi,80)
SourceOrig = 100
OrigDetec = 0

# parallel
ag = AcquisitionGeometry('parallel','2D',angles,detectors)

#cone geometry
#ag = AcquisitionGeometry('cone','2D',angles,detectors,sourcecenter=SourceOrig, centerTodetector=OrigDetec)

# Create ccpi-astra projectir
Aop = AstraProjectorSimple(ig, ag, 'cpu')

#%%

# create sinogram and noisy sinogram
sin = Aop.direct(x)

np.random.seed(1)

noise = 'gaussian'

if noise == 'gaussian':
    noisy_sin = AcquisitionData(sin.as_array() + np.random.normal(0, 2, sin.shape))
elif noise == 'poisson':
    scale = 0.5
    noisy_sin = AcquisitionData(scale * np.random.poisson(sin.as_array()/scale))
    
# simple backprojection
backproj = Aop.adjoint(noisy_sin)

plt.imshow(x.as_array())
plt.title('Phantom image')
plt.show()

plt.imshow(noisy_sin.array)
plt.title('Simulated data')
plt.show()

plt.imshow(backproj.array)
plt.title('Backprojected data')
plt.show()

#%%
alpha = 20 
Grad2D = gradient((ig.voxel_num_x,ig.voxel_num_y))

operator = [ [Grad2D], [Aop] ] 

f = [L1NormOld(Grad2D, alpha), Norm2sq_new(Aop, noisy_sin, c = 0.5, memopt = False)]
g = ZeroFun()


Aop.norm = Aop.get_max_sing_val()
normK = np.sqrt(Grad2D.norm()**2 + Aop.norm**2)
#normK = compute_opNorm(operator)

# Primal & dual stepsizes
sigma = 100
tau = 1/(sigma*normK**2)

opt = {'niter':1000}
res, total_time, its = PDHG_testGeneric(noisy_sin, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)

# Show result
plt.imshow(res.as_array(), cmap = 'viridis')
plt.title('Reconstruction with PDHG')
plt.colorbar()
plt.show()


#%%