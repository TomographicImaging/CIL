#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 19:30:51 2019

@author: evangelos
"""

from ccpi.framework import DataContainer

class ShrinkageOperator():
    
    def __init__(self):
        pass

    def __call__(self, x, tau, out=None):
        
        return x.sign() * (x.abs() - tau).maximum(0) 
   