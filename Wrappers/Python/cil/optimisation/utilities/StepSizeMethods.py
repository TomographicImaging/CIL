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

    """ 
    Applies the Armijo rule to calculate the step size (step_size).

    The Armijo rule runs a while loop to find the appropriate step_size by starting from a very large number (`alpha`). The step_size is found by reducing the step size (by a factor `beta`) in an iterative way until a certain criterion is met. To avoid infinite loops, we add a maximum number of times (`max_iterations`) the while loop is run.

    Parameters
    ----------
    alpha: float, optional, default=1e6
        The starting point for the step size iterations 
    beta: float between 0 and 1, optional, default=0.5
        The amount the step_size is reduced if the criterion is not met
    max_iterations: integer, optional, default is numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
        The maximum number of iterations to find a suitable step size 

    Reference
    ---------
    Algorithm 3.1 (Numerical Optimization, Nocedal, Wright) (https://www.math.uci.edu/~qnie/Publications/NumericalOptimization.pdf)
     https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080

    """

    def __init__(self, alpha=1e6, beta=0.5, max_iterations=None):
        '''Initialises the step size rule 
        '''
        
        self.alpha_orig = alpha
        if self.alpha_orig is None: # Can be removed when alpha and beta are deprecated in GD
            self.alpha_orig = 1e6 

        self.beta = beta 
        if self.beta is None:  # Can be removed when alpha and beta are deprecated in GD
            self.beta = 0.5
            
        self.max_iterations = max_iterations
        if self.max_iterations is None:
            self.max_iterations = numpy.ceil(2 * numpy.log10(self.alpha_orig) / numpy.log10(2))

    def get_step_size(self, algorithm):
        """
        Applies the Armijo rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        k = 0
        self.alpha = self.alpha_orig
        f_x = algorithm.objective_function(algorithm.solution)

        self.x_armijo = algorithm.solution.copy()

        while k < self.max_iterations:

            algorithm.gradient_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.solution.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.objective_function(self.x_armijo)
            sqnorm = algorithm.gradient_update.squared_norm()
            if f_x_a - f_x <= - (self.alpha/2.) * sqnorm:
                break
            k += 1.
            self.alpha *= self.beta

        if k == self.max_iterations:
            raise ValueError(
                'Could not find a proper step_size in {} loops. Consider increasing alpha or max_iterations.'.format(self.max_iterations))
        return self.alpha


class BarzilaiBorweinStepSizeRule(StepSizeRule):

    """ 
    Applies the Barzilai- Borwein rule to calculate the step size (step_size).

    #TODO: Write a short explanation here 
    Parameters
    ----------
    mode: One of 'long', 'short' or 'alternate', default is 'short'. 
        TODO:
    stabilisation_param: float
        TODO:
    

    Reference
    ---------
    Barzilai, Jonathan; Borwein, Jonathan M. (1988). "Two-Point Step Size Gradient Methods". IMA Journal of Numerical Analysis. 8: 141–148
    https://en.wikipedia.org/wiki/Barzilai-Borwein_method
    
    Burdakov, O., Dai, Y.H. and Huang, N., 2019. Stabilized barzilai-borwein method. arXiv preprint arXiv:1907.06409.

    """

    def __init__(self, initial, mode='short', stabilisation_param=2):
        '''Initialises the step size rule 
        '''
 
        self.mode=mode
        self.store_grad=None 
        self.store_x=None
        self.initial=initial
        self.stabilisation_param=stabilisation_param
        
        pass
    

    def get_step_size(self, algorithm):
        """
        Applies the B-B rule to calculate the step size (`step_size`)

        Returns
        --------
        the calculated step size:float

        """
        if self.store_x is None:
            self.store_x=algorithm.x.copy()
            self.store_grad=algorithm.gradient_update.copy()
            return self.initial
        
        if algorithm.gradient_update.norm()<1e-6:
            raise StopIteration

        if self.mode=='long' or (self.mode =='alternate' and algorithm.iteration%2 ==0):
            ret = (( algorithm.x-self.store_x).dot(algorithm.x-self.store_x))/ (( algorithm.x-self.store_x).dot(algorithm.gradient_update-self.store_grad))
        elif self.mode=='short' or (self.mode =='alternate' and algorithm.iteration%2 ==1):
            ret = (( algorithm.x-self.store_x).dot(algorithm.gradient_update-self.store_grad))/ (( algorithm.gradient_update-self.store_grad).dot(algorithm.gradient_update-self.store_grad))
        else:
            raise ValueError('Mode should be chosen from "long", "short" or "alternate". ')
        
        ret = numpy.nanmin( numpy.array([ret, self.stabilisation_param/algorithm.gradient_update.norm()]))
        self.store_x.fill(algorithm.x)
        self.store_grad.fill(algorithm.gradient_update)
        

   
        return ret