# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import ZeroFunction
import numpy
import warnings

class FISTA(Algorithm):
    
    r'''Fast Iterative Shrinkage-Thresholding Algorithm 
    
    Problem : 
    
    .. math::
    
      \min_{x} f(x) + g(x)
    
    |
    
    Parameters :
        
      :param initial: Initial guess ( Default initial = 0)
      :param f: Differentiable function
      :param g: Convex function with " simple " proximal operator


    Reference:
      
        Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. 
        SIAM journal on imaging sciences,2(1), pp.183-202.
    '''
    
    
    def __init__(self, initial=None, f=None, g=ZeroFunction(), use_axpby=True, **kwargs):
        
        '''FISTA algorithm creator 
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up
        
        Optional parameters:

        :param initial: Initial guess ( Default initial = 0)
        :param f: Differentiable function
        :param g: Convex function with " simple " proximal operator'''
        
        super(FISTA, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        self._use_axpby = use_axpby                      

        if initial is not None and f is not None:
            self.set_up(initial=initial, f=f, g=g)

    def set_up(self, initial, f, g=ZeroFunction()):
        '''initialisation of the algorithm

        :param initial: Initial guess ( Default initial = 0)
        :param f: Differentiable function
        :param g: Convex function with " simple " proximal operator'''

        print("{} setting up".format(self.__class__.__name__, ))
        
        self.y = initial.copy()
        self.x_old = initial.copy()
        self.x = initial.copy()
        self.u = initial.copy()

        self.f = f
        self.g = g
        if f.L is None:
            raise ValueError('Error: Fidelity Function\'s Lipschitz constant is set to None')
        self.invL = 1/f.L
        self.t = 1
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

            
    def update(self):
        self.t_old = self.t
        self.f.gradient(self.y, out=self.u)
        self.u.multiply( -self.invL , out=self.u)
        self.u.add( self.y , out=self.u)

        self.g.proximal(self.u, self.invL, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.x.subtract(self.x_old, out=self.y)

        if self._use_axpby:
            self.y.axpby(((self.t_old-1)/self.t), 1, self.x, out=self.y)
        else:
            self.x.substract(self.x_old, out=self.y)
            self.y.multiply((self.t_old-1)/self.t, out=self.y)
            self.y.add(self.x, out=self.y)

        self.x_old.fill(self.x)

        
    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )    
    

