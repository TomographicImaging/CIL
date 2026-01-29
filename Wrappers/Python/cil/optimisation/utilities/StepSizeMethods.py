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
# - CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from abc import ABC, abstractmethod
import numpy
from numbers import Number
import logging

log = logging.getLogger(__name__)

class StepSizeRule(ABC):
    """
    Abstract base class for a step size rule. The abstract method, `get_step_size` takes in an algorithm and thus can access all parts of the algorithm (e.g. current iterate, current gradient, objective functions etc) and from this  should return a float as a step size. 
    """

    def __init__(self):
        '''Initialises the step size rule 
        '''
        pass

    @abstractmethod
    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the calculated step size:float 
        """
        pass


class ConstantStepSize(StepSizeRule):
    """
    Step-size rule that always returns a constant step-size. 

    Parameters
    ----------
    step_size: float
        The step-size to be returned with each call. 
    """

    def __init__(self, step_size):
        '''Initialises the constant step size rule
        
         Parameters:
         -------------
         step_size : float, the constant step size 
        '''
        self.step_size = step_size

    def get_step_size(self, algorithm):
        """
        Returns
        --------
        the calculated step size:float
        """
        return self.step_size


class ArmijoStepSizeRule(StepSizeRule):

    r""" Applies the Armijo rule to calculate the step size (step_size).

    The Armijo rule runs a while loop to find the appropriate step_size by starting from a very large number (`alpha`). The step_size is found by reducing the step size (by a factor `beta`) in an iterative way until a certain criterion is met. To avoid infinite loops, we add a maximum number of times (`max_iterations`) the while loop is run.

    Reference
    ---------
    - Algorithm 3.1 in Nocedal, J. and Wright, S.J. eds., 1999. Numerical optimization. New York, NY: Springer New York. https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)
    
    - https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080
    
    
    Parameters
    ----------
    alpha: float, optional, default=1e6
        The starting point for the step size iterations 
    beta: float between 0 and 1, optional, default=0.5
        The amount the step_size is reduced if the criterion is not met
    max_iterations: integer, optional, default is numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
        The maximum number of iterations to find a suitable step size 
    warmstart: Boolean, default is True
        If `warmstart = True` the initial step size at each Armijo iteration is the calculated step size from the last iteration. If `warmstart = False` at each  Armijo iteration, the initial step size is reset to the original, large `alpha`. 
        In the case of *well-behaved* convex functions, `warmstart = True` is likely to be computationally less expensive. In the case of non-convex functions, or particularly tricky functions, setting `warmstart = False` may be beneficial. 

    """

    def __init__(self, alpha=1e6, beta=0.5, max_iterations=None, warmstart=True):
        '''Initialises the step size rule 
        '''
        
        self.alpha_orig = alpha
        if self.alpha_orig is None: # Can be removed when alpha and beta are deprecated in GD
            self.alpha_orig = 1e6 
        self.alpha = self.alpha_orig
        self.beta = beta 
        if self.beta is None:  # Can be removed when alpha and beta are deprecated in GD
            self.beta = 0.5
            
        self.max_iterations = max_iterations
        if self.max_iterations is None:
            self.max_iterations = numpy.ceil(2 * numpy.log10(self.alpha_orig) / numpy.log10(2))
            
        self.warmstart=warmstart

    def get_step_size(self, algorithm):
        """
        Applies the Armijo rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        k = 0
        if not self.warmstart:  
            self.alpha = self.alpha_orig
        
        f_x = algorithm.calculate_objective_function_at_point(algorithm.solution)

        self.x_armijo = algorithm.solution.copy()
        
        log.debug("Starting Armijo backtracking with initial step size: %f", self.alpha)
        
        while k < self.max_iterations:

            algorithm.gradient_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.solution.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.calculate_objective_function_at_point(self.x_armijo)
            sqnorm = algorithm.gradient_update.squared_norm()
            if f_x_a - f_x <= - (self.alpha/2.) * sqnorm:
                break
            k += 1.
            self.alpha *= self.beta
        
        log.info("Armijo rule took %d iterations to find step size", k)

        if k == self.max_iterations:
            raise ValueError(
                'Could not find a proper step_size in {} loops. Consider increasing alpha or max_iterations.'.format(self.max_iterations))
        
        return self.alpha


class BarzilaiBorweinStepSizeRule(StepSizeRule):

    r""" Applies the Barzilai- Borwein rule to calculate the step size (step_size).

    Let :math:`\Delta x=x_k-x_{k-1}` and :math:`\Delta g=g_k-g_{k-1}`. Where :math:`x_k` is the :math:`k` th iterate (current solution after iteration :math:`k` ) and :math:`g_k` is the gradient calculation in the :math:`k` th iterate, found in :code:`algorithm.gradient_update`.  A Barzilai-Borwein (BB) iteration is :math:`x_{k+1}=x_k-\alpha_kg_k` where the step size :math:`\alpha _k` is either

    - :math:`\alpha_k^{LONG}=\frac{\Delta x\cdot\Delta x}{\Delta x\cdot\Delta g}`, or

    - :math:`\alpha_k^{SHORT}=\frac{\Delta x \cdot\Delta g}{\Delta g \cdot\Delta g}`.
    
    Where the operator :math:`\cdot` is the standard inner product between two vectors. 
    
    This is suitable for use with gradient based iterative methods where the calculated gradient is stored as `algorithm.gradient_update`.
    
    Parameters
    ----------
    initial: float, greater than zero 
        The step-size for the first iteration. We recommend something of the order :math:`1/f.L` where :math:`f` is the (differentiable part of) the objective you wish to minimise.
    mode: One of 'long', 'short' or 'alternate', default is 'short'. 
        This calculates the step-size based on the LONG, SHORT or alternating between the two, starting with short. 
    stabilisation_param: 'auto', float or 'off', default is 'auto'
        In order to add stability the step-size has an upper limit of :math:`\Delta/\|g_k\|` where by 'default', the `stabilisation_param`, :math:`\Delta` is  determined automatically to be the minimium of :math:`\Delta x` from the first 3 iterations. The user can also pass a fixed constant or turn "off" the stabilisation, equivalently passing `np.inf`.
        
    
    Reference
    ---------
    - Barzilai, Jonathan; Borwein, Jonathan M. (1988). "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis. 8: 141–148, https://doi.org/10.1093/imanum/8.1.141
    
    - Burdakov, O., Dai, Y. and Huang, N., 2019. STABILIZED BARZILAI-BORWEIN METHOD. Journal of Computational Mathematics, 37(6). https://doi.org/10.4208/jcm.1911-m2019-0171

    - https://en.wikipedia.org/wiki/Barzilai-Borwein_method
    """

    def __init__(self, initial, mode='short', stabilisation_param="auto"):
        '''Initialises the step size rule 
        '''
 
        self.mode=mode
        if self.mode == 'short':
            self.is_short = True
        elif self.mode == 'long' or self.mode == 'alternate':
            self.is_short = False
        else:
            raise ValueError('Mode should be chosen from "long", "short" or "alternate". ')
        
        self.store_grad=None 
        self.store_x=None
        self.initial=initial
        if stabilisation_param == 'auto':
            self.adaptive = True
            stabilisation_param = numpy.inf
        elif stabilisation_param == "off":
            self.adaptive = False 
            stabilisation_param = numpy.inf
        elif ( isinstance(stabilisation_param, Number) and stabilisation_param >=0):
            self.adaptive = False 
        else:
            raise TypeError(" The stabilisation_param should be 'auto', a positive number or 'off'")
        self.stabilisation_param=stabilisation_param
        
    

    def get_step_size(self, algorithm):
        """
        Applies the B-B rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        #For the first iteration we use an initial step size because the BB step size requires a previous iterate. 
        if self.store_x is None:
            self.store_x=algorithm.x.copy() # We store the last iterate in order to calculate the BB step size 
            self.store_grad=algorithm.gradient_update.copy()# We store the last gradient in order to calculate the BB step size 
            return self.initial
        
        gradient_norm = algorithm.gradient_update.norm()
        #If the gradient is zero, gradient based algorithms will not update and te step size calculation will divide by zero so we stop iterations. 
        if gradient_norm < 1e-8:
            raise StopIteration

        algorithm.x.subtract(self.store_x, out=self.store_x) 
        algorithm.gradient_update.subtract(self.store_grad, out=self.store_grad)
        if self.is_short:
                ret = (self.store_x.dot(self.store_grad))/ (self.store_grad.dot(self.store_grad))
        else:
            ret = (self.store_x.dot(self.store_x))/ (self.store_x.dot(self.store_grad))

        
        #This computes the default stabilisation parameter, using the first three iterations
        if (algorithm.iteration <=3 and self.adaptive):
            self.stabilisation_param = min(self.stabilisation_param, self.store_x.norm() )
        
        # Computes the step size as the minimum of the ret, above, and :math:`\Delta/\|g_k\|` ignoring any NaN values. 
        ret = numpy.nanmin( numpy.array([ret, self.stabilisation_param/gradient_norm]))
        
        # We store the last iterate and gradient in order to calculate the BB step size 
        self.store_x.fill(algorithm.x)
        self.store_grad.fill(algorithm.gradient_update)
        
        if self.mode == "alternate":
            self.is_short =  not self.is_short       
        
        return ret
    
    
class PDHG_strongly_convex_update(StepSizeRule):
    
    
    def __init__(self, gamma_g=None, gamma_fconj=None):
        '''Updates step sizes (theta, sigma, tau) in the PDHG algorithm in the cases of primal or dual acceleration using the strongly convexity property.
        The case where both functions are strongly convex is not available at the moment.
    
        
         Parameters
         -------------
         gamma_g : positive :obj:`float`, optional, default=None
            Strongly convex constant if the function g is strongly convex. Allows primal acceleration of the PDHG algorithm.
         gamma_fconj : positive :obj:`float`, optional, default=None
            Strongly convex constant if the convex conjugate of f is strongly convex. Allows dual acceleration of the PDHG algorithm.
        '''

        if self.gamma_g is not None and self.gamma_fconj is not None:
            raise NotImplementedError("PDHG strongly convex step size update not implemented for both primal and dual acceleration. Please choose only one of gamma_g or gamma_fconj.")
        if isinstance(gamma_g, Number):
            if gamma_g <= 0:
                raise ValueError(
                    "Strongly convex constant is positive, {} is passed for the strongly convex conjugate function of f.".format(gamma_g))
            self.gamma_g = gamma_g
        elif gamma_g is None:
            pass
        else:
            raise ValueError(
                "Positive float is expected for the strongly convex constant of the function g, {} is passed".format(gamma_g))
            pass

        if isinstance(gamma_fconj, Number):
            if gamma_fconj <= 0:
                raise ValueError(
                    "Strongly convex constant is positive, {} is passed for the strongly convex conjugate function of f.".format(gamma_fconj))
            self.gamma_fconj = gamma_fconj
        elif gamma_fconj is None:
            pass
        else:
            raise ValueError(
                "Positive float is expected for the strongly convex constant of the convex conjugate of function f, {} is passed".format(gamma_fconj))
        
    def get_step_size(self, algorithm):
        """
        Applies the PDHG strongly convex step size update to calculate the new primal and dual step sizes

        Returns
        --------
        
        """
        # Update sigma and tau based on the strong convexity of G
        if self.gamma_g is not None:
            theta = 1.0 / np.sqrt(1 + 2 * self.gamma_g * algorithm.tau)
            tau *= theta
            sigma /= theta

        # Update sigma and tau based on the strong convexity of F
        # Following operations are reversed due to symmetry, sigma --> tau, tau -->sigma
        if self.gamma_fconj is not None:
            theta = 1.0 / np.sqrt(1 + 2 * self.gamma_fconj * algorithm.sigma)
            sigma *= theta
            tau /= theta

        return theta, sigma, tau

class PDHG_adaptive_2013(StepSizeRule):
    
    def __init__(self, initial_alpha=0.95, beta=0.95, gamma=0.75, delta=1.5, s=None, eta=0.95, auto_stop=True):
        '''Primal Dual Hybrid Gradient (PDHG) algorithm, see :cite:`CP2011`, :cite:`EZXC2010`.

            Adaptive: https://arxiv.org/pdf/1305.0546
         Parameters
         -------------
        initial_alpha : positive :obj:`float`, optional, default=0.95
            Initial value of the parameter alpha used in the adaptive step size method.
        beta : positive :obj:`float`, optional, default=0.95
            Value of the parameter eta used in the adaptive step size method.
        gamma : positive :obj:`float`, optional, default=0.75
            Value of the parameter c used in the adaptive step size method.
        delta : positive :obj:`float`,greater than one,  optional, default=1.5
            Value of the parameter delta used in the adaptive step size method.
        s : positive :obj:`float`, optional, default= Norm of the operator A 
            Value of the parameter s used in the adaptive step size method.
        eta : positive :obj:`float`, optional, default=0.95
            Value of the parameter eta used in the adaptive step size method.
        auto_stop : :obj:`boolean`, optional, default=True
            If True, the adaptive step size method automatically stops updating the step sizes when they have not changed over five consecutive iterations.
        
        '''
        self.alpha = initial_alpha
        self.eta = eta
        self.beta = beta
        self.delta = delta
        if s is None:
            s = operator.norm()
        self.s = s
        self.gamma = gamma
        self.tolerance = 1e-6
        self.p_norm = 100
        self.d_norm = 100
        
        self.auto_stop = auto_stop
        self.count = 0
        
        self.y_old = None
        self.x_resid = None
        self.y_resid = None
        self.x_store = None
        
        self.adaptive = True
        
    def get_step_size(self, algorithm):
        if self.y_old is None:
            self.y_old = algorithm.y.geometry.allocate(None) # Extra range data 1
            self.x_resid = algorithm.x.geometry.allocate(None) # Extra image 1
            self.y_resid = algorithm.y.geometry.allocate(None) # Extra range data 2
            self.x_store = algorithm.x.geometry.allocate(None) # Extra image 2
        if self.adaptive: 
            if self.p_norm > self.tolerance and self.d_norm > self.tolerance: # adaptive step sizes only when above tolerance 
                #print('Before adaptive', self.tau, self.sigma)
                    b = self._calculate_backtracking()
                    while b>1:
                        algorithm._tau *= self.beta/b
                        algorithm._sigma *= self.beta/b
                        # Swap x and x_store
                        tmp=algorithm.x
                        algorithm.x= algorithm.x_store
                        algorithm.x_store= tmp
                        
                        print('Multiplying step sizes by beta/b, beta = {}, b = {}'.format(self.beta, b))
                        print('tau = {}, sigma = {}'.format( algorithm._tau, algorithm._sigma))
                        algorithm._pdhg_update()
                        b = self._calculate_backtracking()

                    print('After possible reduction', algorithm._tau, algorithm._sigma)
                    algorithm.operator.adjoint(self.y_resid, out=algorithm.x_tmp)
                    algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
                    self.x_resid.sapyb((1/algorithm._tau), algorithm.x_tmp, -1.0, out=algorithm.x_tmp)
                    self.y_resid.sapyb((1/algorithm._sigma), algorithm.y_tmp, -1.0, out=algorithm.y_tmp)
                    self.p_norm = algorithm.x_tmp.norm()
                    self.d_norm = algorithm.y_tmp.norm()
                    if self.p_norm < (self.s/self.delta)*self.d_norm:
                        print('2*self.p_norm < self.d_norm')
                        algorithm._tau *= (1- self.alpha)
                        algorithm._sigma /= (1 - self.alpha)
                        self.alpha *= self.eta
                        self.count = 0
                    elif (self.s*self.delta)*self.d_norm < self.p_norm:
                        print('2*self.d_norm < self.p_norm')
                        algorithm._tau /= (1 - self.alpha)
                        algorithm._sigma *= (1 - self.alpha)
                        self.alpha *=self.eta
                        self.count = 0
                    else:
                        print('No change')
                        self.count += 1
                        pass
                    print('After adaptive', algorithm._tau, algorithm._sigma, self.alpha)
            else:
                    print('No adaptive step size update, below tolerance')
            self.y_old = algorithm.y.copy() # Can i do something other than copying every iteration?  
            
            if self.count>5 and self.auto_stop:
                    self.adaptive = False
                    print('Automatic stopping of adaptive step size updates, step sizes have not changed for 5 iterations')
                    del self.x_resid
                    del self.y_resid
                    del self.x_store
                    del self.y_old
            
            return algorithm._theta, algorithm._tau, algorithm._sigma
        
        
    
    def _calculate_backtracking(self):
        """ Calculates the backtracking parameter b used to update step sizes in the adaptive PDHG algorithm.
            Returns
            -------
            b : :obj:`float`
                Backtracking parameter used to update step sizes in the adaptive PDHG algorithm.
        """
        
        self.x.sapyb(1.0, self.x_old, -1.0, out=self.x_resid) 
        print('self.x, self.x_old = ', self.x.norm(), self.x_old.norm())
        x_change_norm = self.x_resid.norm()
        self.y.sapyb(1.0, self.y_old, -1.0, out=self.y_resid)
        y_change_norm = self.y_resid.norm()
        self.operator.direct(self.x_resid, out=algorithm.y_tmp)
        cross_term = np.abs(2*self.sigma*self.tau*self.y_resid.dot(algorithm.y_tmp))
        print('cross_term = ', cross_term
              , 'x_change_norm = ', x_change_norm, 'y_change_norm = ', y_change_norm)
        b = cross_term/((self.gamma*self.sigma)*x_change_norm**2 + (self.gamma*self.tau)*y_change_norm**2 )
        print(b)
        return b