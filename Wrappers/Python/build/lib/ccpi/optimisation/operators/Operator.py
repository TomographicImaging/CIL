# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:55:56 2019

@author: ofn77899
"""
from ccpi.optimisation.operators.ScaledOperator import ScaledOperator

class Operator(object):
    '''Operator that maps from a space X -> Y'''
    def is_linear(self):
        '''Returns if the operator is linear'''
        return False
    def direct(self,x, out=None):
        '''Returns the application of the Operator on x'''
        raise NotImplementedError
    def norm(self):
        '''Returns the norm of the Operator'''
        raise NotImplementedError
    def range_geometry(self):
        '''Returns the range of the Operator: Y space'''
        raise NotImplementedError
    def domain_geometry(self):
        '''Returns the domain of the Operator: X space'''
        raise NotImplementedError
    def __rmul__(self, scalar):
        '''Defines the multiplication by a scalar on the left

        returns a ScaledOperator'''
        return ScaledOperator(self, scalar)
