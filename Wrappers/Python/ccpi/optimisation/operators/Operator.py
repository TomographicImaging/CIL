# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:55:56 2019

@author: ofn77899
"""
from ccpi.framework,operators import ScaledOperator

class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def __init__(self, **kwargs):
        self.scalar = 1
    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    def direct(self,x, out=None):
        raise NotImplementedError
    def size(self):
        # To be defined for specific class
        raise NotImplementedError
    def norm(self):
        raise NotImplementedError
    def range_geometry(self):
        raise NotImplementedError
    def domain_geometry(self):
        raise NotImplementedError
    def __rmul__(self, scalar):
        return ScaledOperator(self, scalar)
    
