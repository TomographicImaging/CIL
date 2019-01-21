#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 19:44:26 2019

@author: evangelos
"""

from my_changes import *
from ccpi.framework import DataContainer
from ccpi.optimisation.funcs import Identity

data = DataContainer(np.zeros((2,3), 'int64'))
fidelity = L1Norm(Identity(), data, c = 1)

x = DataContainer(np.random.randint(10, size = (2,3)))
fidelity(x)

#%% 