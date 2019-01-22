#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 21:51:21 2019

@author: evangelos
"""

"""Total variation denoising using PDHG.
Solves the optimization problem
    min_{x >= 0}  1/2 ||x - g||_2^2 + lam || |grad(x)| ||_1
Where ``grad`` the spatial gradient and ``g`` is given noisy data.
For further details and a description of the solution method used, see
https://odlgroup.github.io/odl/guide/pdhg_guide.html in the ODL documentation.
"""

import numpy as np
import scipy.misc
import odl

# Read test image: use only every second pixel, convert integer to float,
# and rotate to get the image upright
image = np.rot90(scipy.misc.ascent()[::2, ::2], 3).astype('float')
shape = image.shape

# Rescale max to 1
image /= image.max()

# Discretized spaces
space = odl.uniform_discr([0, 0], shape, shape)

# Original image
orig = space.element(image)

# Add noise
image += 0.1 * odl.phantom.white_noise(orig.space)

# Data of noisy image
noisy = space.element(image)

# Gradient operator
gradient = odl.Gradient(space)

# Matrix of operators
op = odl.BroadcastOperator(odl.IdentityOperator(space), gradient)

#%%