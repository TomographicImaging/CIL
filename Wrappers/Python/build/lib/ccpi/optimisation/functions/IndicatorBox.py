# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Jakob Jorgensen, Daniil Kazantsev and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
from ccpi.optimisation.functions import Function
import numpy

class IndicatorBox(Function):
    '''Box constraints indicator function. 
    
    Calling returns 0 if argument is within the box. The prox operator is projection onto the box. 
    Only implements one scalar lower and one upper as constraint on all elements. Should generalise
    to vectors to allow different constraints one elements.
'''
    
    def __init__(self,lower=-numpy.inf,upper=numpy.inf):
        # Do nothing
        super(IndicatorBox, self).__init__()
        self.lower = lower
        self.upper = upper
        
    
    def __call__(self,x):
        
        if (numpy.all(x.array>=self.lower) and 
            numpy.all(x.array <= self.upper) ):
            val = 0
        else:
            val = numpy.inf
        return val
    
    def prox(self,x,tau=None):
        return  (x.maximum(self.lower)).minimum(self.upper)
    
    def proximal(self, x, tau, out=None):
        if out is None:
            return self.prox(x, tau)
        else:
            if not x.shape == out.shape:
                raise ValueError('Norm1 Incompatible size:',
                                 x.shape, out.shape)
            #(x.abs() - tau*self.gamma).maximum(0) * x.sign()
            x.abs(out = out)
            out.__isub__(tau*self.gamma)
            out.maximum(0, out=out)
            if self.sign_x is None or not x.shape == self.sign_x.shape:
                self.sign_x = x.sign()
            else:
                x.sign(out=self.sign_x)
                
            out.__imul__( self.sign_x )
