#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from numbers import Number
import numpy
from ccpi.optimisation.functions import Function
from ccpi.framework import ImageGeometry, ImageData

class ScaledFunction(Function):

    def __init__(self, function, scalar):
        
#        super(ScaledFunction, self).__init__()
        self.L = None
        if not isinstance (scalar, Number):
            raise TypeError('expected scalar: got {}'.format(type(scalar)))
        self.scalar = scalar
        self.function = function

    def __call__(self,x, out=None):
        
        return self.scalar * self.function(x)

    def convex_conjugate(self, x, out=None):
        
        return self.scalar * self.function.convex_conjugate(x/self.scalar)
    
    def gradient(self, x, out=None):
        
        if out is None:            
            return self.scalar * self.function.gradient(x)    
        else:
            out.fill( self.scalar * self.function.gradient(x) )
        
    def proximal(self, x, tau, out=None):

        if out is None:
            return self.function.proximal(x, tau*self.scalar)     
        else:
            out.fill( self.function.proximal(x, tau*self.scalar) )
                    
    def proximal_conjugate(self, x, tau, out = None):

        if out is None:
            return self.scalar  * self.function.proximal_conjugate(x/self.scalar, tau/self.scalar)
        else:
            out.fill(self.scalar  * self.function.proximal_conjugate(x/self.scalar, tau/self.scalar))

    def grad(self, x):
        '''Alias of gradient(x,None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use gradient instead''', DeprecationWarning)
        return self.gradient(x, out=None)

    def prox(self, x, tau):
        '''Alias of proximal(x, tau, None)'''
        warnings.warn('''This method will disappear in following 
        versions of the CIL. Use proximal instead''', DeprecationWarning)
        return self.proximal(x, out=None)


