# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from ccpi.optimisation.algorithms import Algorithm
from ccpi.optimisation.functions import ZeroFunction
import numpy

class FISTA(Algorithm):
    '''Fast Iterative Shrinkage-Thresholding Algorithm
    
    Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
    algorithm for linear inverse problems. 
    SIAM journal on imaging sciences,2(1), pp.183-202.
    
    Parameters:
      x_init: initial guess
      f: data fidelity
      g: regularizer
      opt: additional options 
    '''
    
    
    def __init__(self, **kwargs):
        '''initialisation can be done at creation time if all 
        proper variables are passed or later with set_up'''
        super(FISTA, self).__init__()
        self.f = kwargs.get('f', None)
        self.g = kwargs.get('g', ZeroFunction())
        self.x_init = kwargs.get('x_init',None)
        self.invL = None
        self.t_old = 1
        if self.x_init is not None and \
           self.f is not None and self.g is not None:
            print ("FISTA set_up called from creator")
            self.set_up(self.x_init, self.f, self.g)        

    
    def set_up(self, x_init, f, g, opt=None, **kwargs):
        
        self.f = f
        self.g = g
        
        # algorithmic parameters
        if opt is None: 
            opt = {'tol': 1e-4}
        
        self.y = x_init.copy()
        self.x_old = x_init.copy()
        self.x = x_init.copy()
        self.u = x_init.copy()            


        self.invL = 1/f.L
        
        self.t_old = 1
        self.update_objective()
        self.configured = True
            
    def update(self):

        self.f.gradient(self.y, out=self.u)
        self.u.__imul__( -self.invL )
        self.u.__iadd__( self.y )
        # x = g.prox(u,invL)
        self.g.proximal(self.u, self.invL, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
#        self.x.subtract(self.x_old, out=self.y)
        self.y = self.x - self.x_old
        self.y.__imul__ ((self.t_old-1)/self.t)
        self.y.__iadd__( self.x )
        
        self.x_old.fill(self.x)
        self.t_old = self.t            
        
    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )    
    

