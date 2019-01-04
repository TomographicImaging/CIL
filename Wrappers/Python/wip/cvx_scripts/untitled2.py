#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  1 19:29:07 2019

@author: evangelos
"""

import numpy as np

def gradient(img):
    '''
    Compute the gradient of an image as a numpy array
    Code from https://github.com/emmanuelle/tomo-tv/
    '''
    shape = [img.ndim, ] + list(img.shape)
    gradient = np.zeros(shape, dtype=img.dtype)
    slice_all = [0, slice(None, -1),]
    for d in range(img.ndim):
        gradient[slice_all] = np.diff(img, axis=d)
        slice_all[0] = d + 1
        slice_all.insert(1, slice(None))
    return gradient

#%%
    
u1 = np.random.randint(10, size = (3,2))
g = gradient(u1)
print(u1)
print(g)