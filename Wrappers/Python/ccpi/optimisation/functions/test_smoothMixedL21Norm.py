#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 10:25:20 2020

@author: evangelos
"""

from ccpi.framework import ImageGeometry, BlockGeometry
from ccpi.optimisation.functions import SmoothMixedL21Norm, MixedL21Norm
import numpy 

ig = ImageGeometry(4, 5)
bg = BlockGeometry(ig, ig)

epsilon = 0.5

f1 = SmoothMixedL21Norm(epsilon)    
x = bg.allocate('random_int')

# check call
res1 = f1(x)
tmp = x.get_item(0) * 0.
for el in x.containers:
    tmp += el.power(2.)
tmp+=epsilon**2        
res2 = tmp.sqrt().sum()
numpy.testing.assert_almost_equal(res1, res2) 

# check gradient

res1 = f1.gradient(x)
res2 = x.divide(tmp.sqrt())
numpy.testing.assert_array_almost_equal(res1.get_item(0).as_array(), 
                                        res2.get_item(0).as_array()) 

numpy.testing.assert_array_almost_equal(res1.get_item(1).as_array(), 
                                        res2.get_item(1).as_array()) 

# check with MixedL21Norm, when epsilon close to 0
f1 = SmoothMixedL21Norm(1e-12)   
f2 = MixedL21Norm()

res1 = f1(x)
res2 = f2(x)
numpy.testing.assert_almost_equal(f1(x), f2(x)) 





#%%