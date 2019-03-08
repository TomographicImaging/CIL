#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 23:36:19 2019

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

from algorithms import PDHG
from operators import CompositeOperator, Identity, CompositeDataContainer
from GradientOperator import Gradient
#from functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq ,CompositeFunction
from test_functions import L1Norm, ZeroFun, mixed_L12Norm, L2NormSq, CompositeFunction

from Sparse_GradMat import GradOper

###############################################################################
N = 500
ig = (N,N)
ag = ig

# Create noisy data. Add Gaussian noise
noisy_data = ImageData(np.random.randint(10, size = ig))
alpha = 1

# Create operators
op1 = Gradient(ig)
op2 = Identity(ig, ag)

operator = CompositeOperator((2,1), op1, op2 ) 

noisy_data = ImageData(np.random.randint(10, size = ig))

# Create composite function
f_CompFunction = CompositeFunction(operator, mixed_L12Norm(alpha), \
                                             L2NormSq(0.5, b = noisy_data) )
# create separate functions
f1 = mixed_L12Norm(alpha, A = op1)
f2 = L2NormSq(0.5, A = op2, b = noisy_data)

# random variable to test objective
x = ImageData(np.random.randint(10, size = ig))

# Check Primal objective
res = f1(x) + f2(x)

#print(res, ImageData(op1.direct(x).power(2).sum(axis=0)).sqrt().sum() +
#              0.5 * (op2.direct(x)-noisy_data).power(2).sum())

print(res, f_CompFunction(CompositeDataContainer(x)))

y = ImageData(np.random.randint(10, size = ((2,)+ig)))


zz = f1.convex_conjugate(y)
tt = f2.convex_conjugate(-1*operator.adjoint(CompositeDataContainer(y)))

#%%





