#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt


from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import ZeroFunction, SGFunction, SVRGFunction, LSVRGFunction, SAGAFunction, SAGFunction, ApproximateGradientSumFunction
import logging
import warnings
class PD3O(Algorithm):
    

    r"""Primal Dual Three Operator Splitting (PD3O) algorithm, see "A New Primal–Dual Algorithm for Minimizing the Sum
        of Three Functions with a Linear Operator".  This is a primal dual algorithm for minimising :math:`f(x)+g(x)+h(Ax)` where all functions are proper, lower semi-continuous and convex, 
        :math:`f` should be differentiable with a Lipschitz continuous gradient and :math:`A` is a bounded linear operator. 
    
        Parameters
        ----------
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A convex function with a computationally computable proximal.
        h : Function
            A composite convex function.
        operator: Operator
            Bounded linear operator
        delta: Float, optional, default is  `1./(gamma*operator.norm()**2)`
            The dual step-size 
        gamma: Float, optional, default is `2.0/f.L`
            The primal step size 
        initial : DataContainer, optional default is a container of zeros, in the domain of the operator 
            Initial point for the  algorithm.             


        Reference
        ---------
        Yan, M. A New Primal–Dual Algorithm for Minimizing the Sum of Three Functions with a Linear Operator. J Sci Comput 76, 1698–1717 (2018). https://doi.org/10.1007/s10915-018-0680-3
     """    


    def __init__(self, f, g, h, operator, delta=None, gamma=None, initial=None, **kwargs):

        super(PD3O, self).__init__(**kwargs)

              
        self.set_up(f=f, g=g, h=h,  operator=operator, delta=delta, gamma=gamma, initial=initial, **kwargs)
 
                  
    def set_up(self, f, g, h, operator, delta=None, gamma=None, initial=None,**kwargs):
        
        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        self.f = f # smooth function
        if isinstance(self.f, ZeroFunction):
            warnings.warn(" If self.f is the ZeroFunction, then PD3O = PDHG. Please use PDHG instead. Otherwise, select a relatively small parameter gamma ", UserWarning)
            if gamma is None:
                gamma = 1.0/operator.norm()                      
        
        self.g = g # proximable
        self.h = h # composite
        self.operator = operator
        
        if gamma is None:
            gamma = 0.99*2.0/self.f.L
        
        if delta is None :
            delta = 0.99/(gamma*self.operator.norm()**2)
        
        self.gamma = gamma
        self.delta = delta  

        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.x_old = self.x.copy()    
        
        self.s_old = self.operator.range_geometry().allocate(0)
        self.s = self.operator.range_geometry().allocate(0)
                
        self.grad_f = self.operator.domain_geometry().allocate(0)        
  
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
        
        # initial proximal conjugate step
        self.operator.direct(self.x_old, out=self.s)
        self.s_old.sapyb(1, self.s, self.delta, out=self.s_old)
        self.h.proximal_conjugate(self.s_old, self.delta, out=self.s)
        

    def update(self):
        r""" Performs a single iteration of the PD3O algorithm        
        """

        # Following equations 4 in https://link.springer.com/article/10.1007/s10915-018-0680-3
        # in this case order of proximal steps we recover the (primal) PDHG, when f=0

        
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp
        
        
        # proximal step        
        self.f.gradient(self.x_old, out=self.grad_f)
        self.x_old.sapyb(1., self.grad_f, -self.gamma, out = self.grad_f) # x_old - gamma * grad_f(x_old)        
        self.operator.adjoint(self.s, out=self.x_old)
        self.x_old.sapyb(-self.gamma, self.grad_f, 1.0, out=self.x_old)
        self.g.proximal(self.x_old, self.gamma, out = self.x)
    
        # update step        
        
        if isinstance(self.f, (SVRGFunction, LSVRGFunction)):
            if len(self.f.data_passes_indices[-1]) == self.f.sampler.num_indices:
                self.f._update_full_gradient_and_return(self.x, out=self.x_old)
            else:
                self.f.approximate_gradient( self.x, self.f.function_num, out=self.x_old)
            self.f._data_passes_indices.pop(-1)    
        elif isinstance(self.f, ApproximateGradientSumFunction):
            self.f.approximate_gradient( self.x, self.f.function_num, out=self.x_old)
        else:
            self.f.gradient(self.x, out=self.x_old)    
            
                    
        self.x_old *= self.gamma
        self.grad_f += self.x_old
        self.x.sapyb(2, self.grad_f, -1.0,  out=self.x_old) # 2*x - x_old + gamma*(grad_f_x_old) - gamma*(grad_f_x)
        
        tmp = self.s_old
        self.s_old = self.s
        self.s = tmp
        
        # proximal conjugate step
        self.operator.direct(self.x_old, out=self.s)
        self.s_old.sapyb(1, self.s, self.delta, out=self.s_old)
        self.h.proximal_conjugate(self.s_old, self.delta, out=self.s)
        
        
               
           
                                                                        
    def update_objective(self):
        """
        Evaluates the primal objective
        """
        self.operator.direct(self.x, out=self.s_old)        
        fun_h = self.h(self.s_old)         
        fun_g = self.g(self.x)
        fun_f = self.f(self.x)
        p1 = fun_f + fun_g + fun_h
        
        self.loss.append(p1)
        
        
        