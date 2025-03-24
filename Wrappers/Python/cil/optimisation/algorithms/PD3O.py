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
from cil.optimisation.functions import ZeroFunction, ScaledFunction, SGFunction, SVRGFunction, LSVRGFunction, SAGAFunction, SAGFunction, ApproximateGradientSumFunction
import logging
import warnings

class FunctionWrappingForPD3O():  
    """
    An internal class that wraps the functions, :math:`f`, in PD3O to allow :math:`f` to be an `ApproximateGradientSumFunction`. 
    
    Note that currently :math:`f`  can be any type of deterministic function, an `ApproximateGradientSumFunction` or a scaled `ApproximateGradientSumFunction` but this is not set up to work for a `SumFunction` or `TranslatedFunction` which contains `ApproximateGradientSumFunction`s. 
    
    Parameters
    ----------
    f : function 
        The function :math:`f` to use in PD3O
    """
    def __init__(self, f):  
        self.f = f 
        self.scalar = 1
        while isinstance(self.f, ScaledFunction):
            self.scalar *= self.f.scalar
            self.f = self.f.function
            
        self._gradient_call_index = 0
        
        if isinstance(self.f, (SVRGFunction, LSVRGFunction)):
            self.gradient = self.svrg_gradient
        elif isinstance(self.f, ApproximateGradientSumFunction):  
            self.gradient = self.approximate_sum_function_gradient
        else:
            self.gradient = f.gradient
        
    def svrg_gradient(self, x, out=None):
        if self._gradient_call_index == 0:  
            self._gradient_call_index += 1  
            self.f.gradient(x, out) 
        else: 
            if len(self.f.data_passes_indices[-1]) == self.f.sampler.num_indices:  
                self.f._update_full_gradient_and_return(x, out=out)  
            else:  
                self.f.approximate_gradient( x, self.f.function_num, out=out)  
            self.f._data_passes_indices.pop(-1)   
            self._gradient_call_index = 0  
            
        if self.scalar != 1:
            out *= self.scalar
        return out
    
    def approximate_sum_function_gradient(self, x, out=None):
        if self._gradient_call_index == 0:  
            self._gradient_call_index += 1  
            self.f.gradient(x, out) 
        else: 
            self.f.approximate_gradient( x, self.f.function_num, out=out) 
            self._gradient_call_index = 0  
             
        if self.scalar != 1:
            out *= self.scalar
        return out
    

    def __call__(self, x):  
        return self.scalar * self.f(x)  
    
    @property
    def L(self):
        return  abs(self.scalar) * self.f.L
    
    
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

        Note
        -----
        Note that currently :math:`f` in PD3O can be any type of deterministic function, an `ApproximateGradientSumFunction` or a scaled `ApproximateGradientSumFunction` but we have not implemented or tested when it is a `SumFunction` or `TranslatedFunction` which contains `ApproximateGradientSumFunction`s. 

        Reference
        ---------
        Yan, M. A New Primal–Dual Algorithm for Minimizing the Sum of Three Functions with a Linear Operator. J Sci Comput 76, 1698–1717 (2018). https://doi.org/10.1007/s10915-018-0680-3
     """    


    def __init__(self, f, g, h, operator, delta=None, gamma=None, initial=None, **kwargs):

        super(PD3O, self).__init__(**kwargs)

              
        self.set_up(f=f, g=g, h=h,  operator=operator, delta=delta, gamma=gamma, initial=initial, **kwargs)
 
                  
    def set_up(self, f, g, h, operator, delta=None, gamma=None, initial=None,**kwargs):
        
        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        
        
        if isinstance(f, ZeroFunction):
            warnings.warn(" If f is the ZeroFunction, then PD3O = PDHG. Please use PDHG instead. Otherwise, select a relatively small parameter gamma ", UserWarning)
            if gamma is None:
                gamma = 1.0/operator.norm()                      
        
        self.f = FunctionWrappingForPD3O(f) 
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
        
        
        