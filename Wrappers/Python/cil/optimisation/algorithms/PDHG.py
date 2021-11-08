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
import warnings
import numpy as np


class PDHG(Algorithm):

    r"""Primal Dual Hybrid Gradient (PDHG) algorithm, see :cite:`CP2011`, :cite:`EZXC2010`.

    A first-order primal-dual algorithm for convex optimization problems with known saddle-point structure with applications in imaging. 

    The general problem considered in the PDHG algorithm is the generic saddle-point problem 

    .. math:: \min_{x\in X}\max_{y\in Y} \langle Kx, y \rangle + g(x) - f^{*}(x)

    where :math:`f` and :math:`g` are convex functions with "simple" proximal operators. 
    
    :math:`X` and :math:`Y` are two two finite-dimensional vector spaces with an inner product and representing the domain of :math:`g` and :math:`f^{*}`, the convex conjugate of :math:`f`, respectively.

    The operator :math:`K` is a continuous linear operator with operator norm defined as 

    .. math:: \|K\| = \max\{ \|Kx\| : x\in X, \|x\|\leq1\}


    The saddle point problem is decomposed into the primal problem:

    .. math:: \min_{x\in X} f(Kx) + g(x), 

    and its corresponding dual problem 

    .. math:: \max_{y\in Y} - g^{*}(-K^{*}y) - f^{*}(y).

    The PDHG algorithm consists of three steps:

        a) gradient ascent step for the dual problem,
        b) gradient descent step for the primal problem and
        c) an over-relaxation of the primal variable.


    Notes
    -----    

        - Convergence is guaranteed if the operator norm :math:`\|K\|`, \the dual step size :math:`\sigma` and the primal step size :math:`\tau`, satisfy the following inequality:

        .. math:: 
    
            \tau \sigma \|L\|^2 < 1

        - By default, the step sizes :math:`\sigma` and :math:`\tau` are:

        .. math::

            \sigma = \frac{1}{\|K\|},  \tau = \frac{1}{\|K\|}

        - PDHG algorithm can be accelerated if the functions :math:`f^{*}` and/or :math:`g` are strongly convex.
        
            A function :math:`f` is strongly convex with constant :math:`\gamma>0` if

            .. math::

                f(x) - \frac{\gamma}{2}\|x\|^{2}

            is convex. 
            
            For instance the :math:`\frac{1}{2}\|x\|^{2}_{2}` is :math:`\gamma` strongly convex\
            for :math:`\gamma\in(-\infty,1]`. We say it is 1-strongly convex because it is the largest constant for which \
            :math:`f - \frac{1}{2}\|\cdot\|^{2}` is convex.

            The :math:`\|\cdot\|_{1}` norm is not strongly convex.

            For more information, see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions    

    Example
    -------
    Least Squares minimisation with PDHG.

    .. math:: \min_{u\in X} \|A u - g\|^{2}

    >>> operator = A
    >>> f = L2NormSquared(b = g)
    >>> g = ZeroFunction()
    >>> pdhg = PDHG(f = f, g = g, operator = operator)
    >>> pdhg.run(10)

    Example 
    -------
    Total variation denoising with with PDHG.  

    .. math:: \min_{x\in X} \|u - g\|^{2} + \alpha\|\nabla u\|_{2,1}

    >>> ig = g.geometry
    >>> operator = GradientOperator(ig)
    >>> f = MixedL21Norm()
    >>> g = L2NormSquared(b=g)
    >>> pdhg = PDHG(f = f, g = g, operator = operator)
    >>> pdhg.run(10) 

    References
    ----------

    .. bibliography::    
            

    """

    def __init__(self, f=None, g=None, operator=None, tau=None, sigma=1.,initial=None, use_axpby=True, **kwargs):
        '''PDHG algorithm creator

        Optional parameters

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal of its conjugate. 
        :param g: Convex function with "simple" proximal .
        :param tau: Step size parameter for Primal problem
        :param sigma: Step size parameter for Dual problem
        :param initial: Initial guess ( Default initial = 0)
        :param gamma_g: Strongly convex constant for the function g.     
        :param gamma_fconj: Strongly convex constant for the convex conjugate of the function f.

        '''
        super(PDHG, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        self._use_axpby = use_axpby

        if f is not None and operator is not None and g is not None:
            self.set_up(f=f, g=g, operator=operator, tau=tau, sigma=sigma, initial=initial, **kwargs)

    def set_up(self, f, g, operator, tau=None, sigma=1., initial=None, **kwargs):
        '''initialisation of the algorithm

        :param operator: a Linear Operator
        :param f: Convex function with "simple" proximal of its conjugate. 
        :param g: Convex function with "simple" proximal 
        :param tau: Step size parameter for Primal problem
        :param sigma: Step size parameter for Dual problem
        :param initial: Initial guess ( Default initial = 0)'''

        print("{} setting up".format(self.__class__.__name__, ))
        
        # can't happen with default sigma
        if sigma is None and tau is None:
            raise ValueError('Need sigma*tau||K||^2<1')
        # algorithmic parameters
        self.f = f
        self.g = g
        self.operator = operator

        normK = self.operator.norm()
        self.tau = 1./normK
        self.sigma = 1./normK
        
        if initial is None:
            self.x_old = self.operator.domain_geometry().allocate(0)
        else:
            self.x_old = initial.copy()

        self.x = self.x_old.copy()
        self.x_tmp = self.operator.domain_geometry().allocate(0)
        self.y = self.operator.range_geometry().allocate(0)
        self.y_tmp = self.operator.range_geometry().allocate(0)   

        # relaxation parameter, default value is 1.0
        self.theta = kwargs.get('theta',1.0)

        # Primal Acceleration: Function g is strongly convex 
        self.gamma_g = kwargs.get('gamma_g', None)

        # Dual Acceleration : Convex conjugate of f is strongly convex
        self.gamma_fconj = kwargs.get('gamma_fconj', None) 

        try:
            self.gamma_g = self.g.gamma
        except AttributeError:
            pass

        try:
            self.gamma_fconj = self.f.conjugate.gamma
        except AttributeError:
            pass        

        if self.gamma_g is not None:
            warnings.warn("Primal Acceleration of PDHG: The function g is assumed to be strongly convex with parameter `gamma_g`. Need to be sure that gamma_g = {} is the correct strongly convex constant for g. ".format(self.gamma_g))
        
        if self.gamma_fconj is not None:
            warnings.warn("Dual Acceleration of PDHG: The convex conjugate of function f is assumed to be strongly convex with parameter `gamma_fconj`. Need to be sure that gamma_fconj = {} is the correct strongly convex constant".format(self.gamma_fconj))
        

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

    def update_previous_solution(self):
        # swap the pointers to current and previous solution
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def get_output(self):
        # returns the current solution
        return self.x_old

    def update(self):

        #calculate x-bar and store in self.x_tmp
        if self._use_axpby:
            self.x_old.axpby((self.theta + 1.0), -self.theta , self.x, out=self.x_tmp) 
        else:
            self.x_old.subtract(self.x, out=self.x_tmp)
            self.x_tmp *= self.theta
            self.x_tmp += self.x_old

        # Gradient ascent for the dual variable
        self.operator.direct(self.x_tmp, out=self.y_tmp)
        
        if self._use_axpby:
            self.y_tmp.axpby(self.sigma, 1.0 , self.y, out=self.y_tmp)
        else:
            self.y_tmp *= self.sigma
            self.y_tmp += self.y

        self.f.proximal_conjugate(self.y_tmp, self.sigma, out=self.y)

        # Gradient descent for the primal variable
        self.operator.adjoint(self.y, out=self.x_tmp)

        if self._use_axpby:
            self.x_tmp.axpby(-self.tau, 1.0 , self.x_old, self.x_tmp)
        else:
            self.x_tmp *= -1.0*self.tau
            self.x_tmp += self.x_old

        self.g.proximal(self.x_tmp, self.tau, out=self.x)

        #update_previous_solution() called after update by base class
        #i.e current solution is now in x_old, previous solution is now in x        

        # update the step sizes for special cases
        self.update_step_sizes()


    def update_step_sizes(self):
    
        # Update sigma and tau based on the strong convexity of G
        if self.gamma_g is not None:
            self.theta = 1.0/ np.sqrt(1 + 2 * self.gamma_g * self.tau)
            self.tau *= self.theta
            self.sigma /= self.theta 

        # Update sigma and tau based on the strong convexity of F
        # Following operations are reversed due to symmetry, sigma --> tau, tau -->sigma
        if self.gamma_fconj is not None:            
            self.theta = 1.0 / np.sqrt(1 + 2 * self.gamma_fconj * self.sigma)
            self.sigma *= self.theta
            self.tau /= self.theta    

        if self.gamma_g is not None and self.gamma_fconj is not None:
            raise NotImplementedError("This case is not implemented")
                    
        
    def update_objective(self):

        self.operator.direct(self.x_old, out=self.y_tmp)
        f_eval_p = self.f(self.y_tmp)
        g_eval_p = self.g(self.x_old)
        p1 = f_eval_p + g_eval_p

        self.operator.adjoint(self.y, out=self.x_tmp)
        self.x_tmp.multiply(-1.0, out=self.x_tmp)

        f_eval_d = self.f.convex_conjugate(self.y)
        g_eval_d = self.g.convex_conjugate(self.x_tmp)
        d1 = f_eval_d + g_eval_d

        self.loss.append([p1, -d1, p1+d1])
        
    @property
    def objective(self):
        '''alias of loss'''
        return [x[0] for x in self.loss]

    @property
    def dual_objective(self):
        return [x[1] for x in self.loss]
    
    @property
    def primal_dual_gap(self):
        return [x[2] for x in self.loss]

