#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reading multi-channel data and reconstruction using FISTA modular
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
#sys.path.append('../../../data/')
from read_IPdata import read_IPdata

from ccpi.reconstruction.astra_ops import AstraProjector
from ccpi.reconstruction.funcs import Norm2sq , BaseFunction
from ccpi.reconstruction.algs import FISTA
#from ccpi.reconstruction.funcs import BaseFunction

from ccpi.framework import VolumeData, SinogramData


from ccpi.filters.cpu_regularizers_boost import SplitBregman_TV , FGP_TV ,\
                                                 LLT_model, PatchBased_Regul ,\
                                                 TGV_PD


# TV regularization class for FGP_TV method
class TV_FGP(BaseFunction):
    def __init__(self,lambdaReg,iterationsTV):
        # set parameters
        self.lambdaReg = lambdaReg
        self.iterationsTV = iterationsTV
        super(TV_FGP, self).__init__()
    def fun(self,x):
        # function to calculate energy from utils can be used here
        return 0
    def prox(self,x,Lipshitz):
        pars = {'algorithm' : FGP_TV , \
                'input' : x.as_array(),
                'regularization_parameter':self.lambdaReg*Lipshitz, \
                'number_of_iterations' :self.iterationsTV ,\
                'tolerance_constant':1e-4,\
                'TV_penalty': 0}
        
        out = FGP_TV (pars['input'], 
                      pars['regularization_parameter'],
                      pars['number_of_iterations'],
                      pars['tolerance_constant'], 
                      pars['TV_penalty'])
        return out[0]

# read IP paper data into a dictionary
dataDICT = read_IPdata()

# Set ASTRA Projection-backprojection class (fan-beam geometry)
DetWidth = dataDICT.get('im_size')[0] * dataDICT.get('det_width')[0] / \
            dataDICT.get('detectors_numb')[0]
SourceOrig = dataDICT.get('im_size')[0] * dataDICT.get('src_to_rotc')[0] / \
            dataDICT.get('dom_width')[0]
OrigDetec = dataDICT.get('im_size')[0] * \
            (dataDICT.get('src_to_det')[0] - dataDICT.get('src_to_rotc')[0]) /\
            dataDICT.get('dom_width')[0]

# Set up ASTRA projector
Aop = AstraProjector(DetWidth, dataDICT.get('detectors_numb')[0], SourceOrig, 
                     OrigDetec, (np.pi/180)*dataDICT.get('theta')[0], 
                     dataDICT.get('im_size')[0],'fanbeam','gpu') 
# initiate a class object

sino = dataDICT.get('data_norm')[0][:,:,34] # select mid-channel 
b = SinogramData(sino)
# Try forward and backprojection
#backprj = Aop.adjoint(b)

# Create least squares object instance with projector and data.
f = Norm2sq(Aop,b,c=0.5)

# Initial guess
x_init = VolumeData(np.zeros((dataDICT.get('im_size')[0],
                              dataDICT.get('im_size')[0])))

# Run FISTA for least squares without regularization
opt = {'tol': 1e-4, 'iter': 50}
x_fista0, it0, timing0, criter0 = FISTA(x_init, f, None, opt)

plt.imshow(x_fista0.array)
plt.show()

"""
# Now least squares plus 1-norm regularization
#lam = 1
#g0 = Norm1(lam)
# using FGP_TV regularizer

g0 = TV_FGP(lambdaReg = 0.01,iterationsTV=50)

# Run FISTA for least squares plus 1-norm function.
opt = {'tol': 1e-4, 'iter': 50}
x_fista1, it1, timing1, criter1 = FISTA(x_init, f, g0, opt)

plt.imshow(x_fista1.array)
plt.show()
"""