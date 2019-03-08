#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from ccpi.framework import DataContainer, ImageData, ImageGeometry, AcquisitionData
from ccpi.astra.processors import AstraForwardProjector, AstraBackProjector
from ccpi.optimisation.ops import PowerMethodNonsquare
from numbers import Number
import functools

###############################################################################


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
    def allocate_direct(self):
        '''Allocates memory on the Y space'''
        raise NotImplementedError
    def allocate_adjoint(self):
        '''Allocates memory on the X space'''
        raise NotImplementedError
    def range_dim(self):
        raise NotImplementedError
    def domain_dim(self):
        raise NotImplementedError
    def __rmul__(self, other):
        assert isinstance(other, Number)
        self.scalar = other
        return self    
    
class LinearOperator(Operator):
    '''Operator that maps from a space X -> Y'''
    def is_linear(self):
        '''Returns if the operator is linear'''
        return True
    def adjoint(self,x, out=None):
        raise NotImplementedError 