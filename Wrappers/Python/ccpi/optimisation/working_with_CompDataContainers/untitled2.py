#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 20:34:22 2019

@author: evangelos
"""



from ccpi.framework import ImageData, ImageGeometry, \
                           AcquisitionGeometry, AcquisitionData
from ccpi.optimisation.ops import PowerMethodNonsquare


import numpy as np
from numpy import inf
import numpy
import matplotlib.pyplot as plt
from cvxpy import *
from skimage.util import random_noise
import scipy.misc
from skimage.transform import resize

from algorithms import PDHG, PDHG_Composite
from operators import CompositeOperator, Identity, CompositeDataContainer
from GradientOperators import Gradient
from functions import L1Norm, ZeroFun, L2NormSq, CompositeFunction, mixed_L12Norm

from Sparse_GradMat import GradOper

import math


###############################################################################


# Load Data
ig = (2,3)



G = Gradient(ig)
u = ImageData(np.random.randint(10, size=ig))

#f = L_1_2norm(G, 1)


# Check features for L_1_2_norm
# call should return the value of \\Ax||_{1,2} norm
#y = G.direct(u)
#assert math.isclose(ImageData(np.sqrt(y.power(2).sum(axis=0))).sum(), f(u), rel_tol=0.02)


# Check features for L2NormSq
#f = L2NormSq(G)
#print(f(u))

#y = G.direct(u)
#np.sum(np.sqrt(y.as_array()[0]**2 + y.as_array()[1]**2)**2)

#%%

f1 = L1Norm(G)

print(f1(u))

y1 = G.direct(u)
print(np.sum(np.sqrt(y1.as_array()[0]**2 + y1.as_array()[1]**2)))


#print( (y1 * y1).as_array())
#print( (y1 * y1).as_array())
#print( (y1 * y1).as_array().sum())
#%%

#%%







#%%







#%%

    
    