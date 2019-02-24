#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  2 15:29:22 2019

@author: evangelos
"""

import time

from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.funcs import Norm2sq, Norm1, IndicatorBox
from ccpi.optimisation.ops import Operator
from ccpi.optimisation.ops import PowerMethodNonsquare,TomoIdentity

import numpy as np
import matplotlib.pyplot as plt
from cvxpy import *
from my_changes import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

import sys
sys.path.insert(0, '/Users/evangelos/Desktop/Projects/CCPi/CCPi-Framework/Wrappers/Python/wip/cvx_scripts/')
from cvx_functions import *

#%%

data = scipy.misc.face()
data = data/np.max(data)
data1 = resize(data, [200, 300, 3], order=1, mode='reflect', cval=0, clip=True, anti_aliasing=True)
N, M, K = data1.shape
#%%
ig = ImageGeometry(voxel_num_x=K, voxel_num_y = M, channels = N)

noisy_data = ImageData(data1 + 0.25 * np.random.random_sample(data1.shape), geometry=ig)
operator = form_Operator(gradient(ig), TomoIdentity(ig))
alpha = 0.05
f = [TV(alpha), Norm2sq_new(TomoIdentity(ig), noisy_data, c = 0.5, memopt = False)]
plt.imshow(noisy_data.as_array())
plt.show()
g = ZeroFun()

normK = compute_opNorm(operator)
# Primal & dual stepsizes
sigma = 20
tau = 1/(sigma*normK**2)
#sigma = 1.0/normK
#tau = 1.0/normK


#%%

ag = ig
opt = {'niter':1000, 'show_iter':100, 'stop_crit': cmp_L2norm,\
       'tol':1e-5}
res, total_time, its = PDHG(noisy_data, f, g, operator, \
                                  ig, ag, tau = tau, sigma = sigma, opt = opt)

plt.imshow(res.as_array())
plt.show()
