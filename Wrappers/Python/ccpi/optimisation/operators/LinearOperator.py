# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 15:57:52 2019

@author: ofn77899
"""

from ccpi.optimisation.operators import Operator


class LinearOperator(Operator):
    '''A Linear Operator that maps from a space X <-> Y'''
    def __init__(self):
        super(LinearOperator, self).__init__()
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        '''returns the adjoint/inverse operation
        
        only available to linear operators'''
        raise NotImplementedError
