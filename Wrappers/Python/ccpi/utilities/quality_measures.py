#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:04:24 2020

@author: evangelos
"""

from ccpi.optimisation.functions import L2NormSquared, L1Norm
import numpy as np


def mse(dc1, dc2):    
    
    ''' Returns the Mean Squared error of two DataContainers
    '''
    
    diff = dc1 - dc2    
    return L2NormSquared().__call__(diff)/dc1.size()


def mae(dc1, dc2):
    
    ''' Returns the Mean Absolute error of two DataContainers
    '''    
    
    diff = dc1 - dc2  
    return L1Norm().__call__(diff)/dc1.size()

def psnr(ground_truth, corrupted, data_range = 255):

    ''' Returns the Peak signal to noise ratio
    '''   
    
    tmp_mse = mse(ground_truth, corrupted)
    if tmp_mse == 0:
        return 1e5
    return 10 * np.log10((data_range ** 2) / tmp_mse)




