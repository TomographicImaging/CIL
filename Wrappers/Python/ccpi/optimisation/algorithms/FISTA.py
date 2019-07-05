# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
import numpy

class FISTA(Algorithm):
    
    r'''Fast Iterative Shrinkage-Thresholding Algorithm 
    
    Problem : 
    
    .. math::
    
      \min_{x} f(x) + g(x)
    
    |
    
    Parameters :
        
      :parameter x_init : Initial guess ( Default x_init = 0)
      :parameter f : Differentiable function
      :parameter g : Convex function with " simple " proximal operator


    Reference:
      
        Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. 
        SIAM journal on imaging sciences,2(1), pp.183-202.
    '''
    
    
    def __init__(self, **kwargs):
        
        '''creator 
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        
        super(FISTA, self).__init__()
        f = kwargs.get('f', None)
        g = kwargs.get('g', ZeroFunction())
        x_init = kwargs.get('x_init', None)

        if x_init is not None and f is not None:
            print(self.__class__.__name__ , "set_up called from creator")
            self.set_up(x_init=x_init, f=f, g=g)

    def set_up(self, x_init, f, g=ZeroFunction()):

        self.y = x_init.copy()
        self.x_old = x_init.copy()
        self.x = x_init.copy()
        self.u = x_init.copy()

        self.f = f
        self.g = g

        self.invL = 1/f.L
        self.t = 1
        self.update_objective()
        self.configured = True
            
    def update(self):
        self.t_old = self.t
        self.f.gradient(self.y, out=self.u)
        self.u.__imul__( -self.invL )
        self.u.__iadd__( self.y )

        self.g.proximal(self.u, self.invL, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.y = self.x - self.x_old
        self.y.__imul__ ((self.t_old-1)/self.t)
        self.y.__iadd__( self.x )
        
        self.x_old.fill(self.x)

        
    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )    
    

