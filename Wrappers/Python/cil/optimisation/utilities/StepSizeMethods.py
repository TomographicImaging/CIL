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
    Abstract base class for a step size rule. The abstract method, `__call__` takes in an algorithm and thus can access all parts of the algorithm (e.g. current iterate, current gradient, objective functions etc) and from this  should return an  a float as a step size. 
    """
    
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, algorithm):
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
        self.step_size=step_size
        

    def __call__(self, algorithm):

        return self.step_size 


class ArmijoStepSize(StepSizeRule):

    """ 
    Applies the Armijo rule to calculate the step size (step_size).
    
    The Armijo rule runs a while loop to find the appropriate step_size by starting rom a very large number (`alpha`). The step_size is found by reducing the step size (by a factor `beta`) in an iterative way until a certain criterion is met. To avoid infinite loops, we add a maximum number of times (`kmax`) the while loop is run.
        
    Parameters
    ----------
    alpha: float, optional, default=1e6
        The starting point for the step size iterations 
    beta: float between 0 and 1, optional, default=0.5
        The amount the step_size is reduced if the criterion is not met
    kmax: integer, optional, default is numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
        The maximum number of iterations to find a suitable step size 
        
    Reference
    ---------
    Algorithm 3.1 (Numerical Optimization, Nocedal, Wright)
     https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080
    
    """     
    def __init__(self, alpha=1e6, beta=0.5, kmax=None):
        
        '''Initialises the step size rule 

        '''
        self.alpha_orig=alpha
        
        self.beta=beta
        self.kmax=kmax
        if self.kmax is None:
            self.kmax = numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
                
    def __call__(self, algorithm): 

        """
        Applies the Armijo rule to calculate the step size (step_size)
        """
        k = 0
        self.alpha=self.alpha_orig
        f_x = algorithm.objective_function(algorithm.x)
        
        self.gradient_update = algorithm.gradient_update
        
        self.x_armijo = algorithm.x.copy()

        while k < self.kmax:
            # self.x - alpha * self.gradient_update
            self.gradient_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.x.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.objective_function(self.x_armijo)
            sqnorm = self.gradient_update.squared_norm()
            if f_x_a - f_x <= - ( self.alpha/2. ) * sqnorm:
                break
            k += 1.
            self.alpha *= self.beta

        if k == self.kmax:
            raise ValueError('Could not find a proper step_size in {} loops. Consider increasing alpha or kmax.'.format(self.kmax))
        return self.alpha
     
