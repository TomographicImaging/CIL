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
import logging

class PD3O(Algorithm):
    

    r"""Primal Dual Three Operator Splitting (PD3O) algorithm, see "A New Primal–Dual Algorithm for Minimizing the Sum
        of Three Functions with a Linear Operator"
    
        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. 
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A proximable convex function.
        h : Function
            A composite convex function.            

     """    


    def __init__(self, f, g, h, operator, delta=None, gamma=None, initial=None, **kwargs):

        super(PD3O, self).__init__(**kwargs)

        self.f = f # smooth function
        if isinstance(self.f, ZeroFunction):
            logging.warning(" If self.f is the ZeroFunction, then PD3O = PDHG. Please use PDHG instead. Otherwhise, select a relatively small parameter gamma")                        
        
        self.g = g # proximable
        self.h = h # composite
        self.operator = operator
        
        self.delta = delta
        self.gamma = gamma         
        self.set_up(f=f, g=g, operator=operator, delta=delta, gamma=gamma, initial=initial, **kwargs)
 
                  
    def set_up(self, f, g, operator, tau=None, sigma=None, initial=None, **kwargs):

        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        self.x_bar = self.x.copy()    
        self.x_old = self.operator.domain_geometry().allocate(0)
        
        self.s_old = self.operator.range_geometry().allocate(0)
        self.s = self.operator.range_geometry().allocate(0)
                
        self.grad_f = self.operator.domain_geometry().allocate(0)        
  
        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))
        

    def update(self):
        r""" Performs a single iteration of the PD3O algorithm        
        """
        
        # following equations 4 in https://link.springer.com/article/10.1007/s10915-018-0680-3
        # in this case order of proximal steps we recover the (primal) PDHG, when f=0
        # #TODO if we change the order of proximal steps we recover the PDDY algorithm (dual) PDHG, when f=0
        
        # proximal conjugate step
        self.operator.direct(self.x_bar, out=self.s)
        self.s_old.sapyb(1, self.s, self.delta, out=self.s_old)
        self.h.proximal_conjugate(self.s_old, self.delta, out=self.s)
        
        # proximal step        
        self.f.gradient(self.x_old, out=self.grad_f)
        self.x_old.sapyb(1., self.grad_f, -self.gamma, out = self.grad_f) # x_old - gamma * grad_f(x_old)        
        self.operator.adjoint(self.s, out=self.x_old)
        self.x_old.sapyb(-self.gamma, self.grad_f, 1.0, out=self.x_old)
        self.g.proximal(self.x_old, self.gamma, out = self.x)

        # update step        
        self.f.gradient(self.x, out=self.x_bar)                    
        self.x_bar *= self.gamma
        self.grad_f += self.x_bar
        self.x.sapyb(2, self.grad_f, -1.0,  out=self.x_bar) # 2*x - x_old + gamma*(grad_f_x_old) - gamma*(grad_f_x)
        
        self.x_old.fill(self.x)      
        self.s_old.fill(self.s)         
                                                                        
    def update_objective(self):
        """
        Evaluates the primal objective
        """        
                 
        fun_h = self.h(self.operator.direct(self.x))
        fun_g = self.g(self.x)
        fun_f = self.f(self.x)
        p1 = fun_f + fun_g + fun_h
        
        self.loss.append(p1)
        
        
        