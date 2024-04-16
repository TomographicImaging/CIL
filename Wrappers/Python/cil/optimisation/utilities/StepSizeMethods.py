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


class StepSizeMethod(ABC):
    
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, algorithm):
        pass

class ConstantStepSize(StepSizeMethod):
    
    def __init__(self, step_size):
        self.step_size=step_size
        

    def __call__(self, algo):

        return self.step_size 


class ArmijoStepSize(StepSizeMethod):

    """ Algorithm 3.1 (Numerical Optimization, Nocedal, Wright)
    Satisfies armijo condition
    """     
    def __init__(self, alpha=1e6, beta=0.5, kmax=None):
        
        '''Applies the Armijo rule to calculate the step size (step_size)

        https://projecteuclid.org/download/pdf_1/euclid.pjm/1102995080

        The Armijo rule runs a while loop to find the appropriate step_size by starting
        from a very large number (alpha). The step_size is found by dividing alpha by 2
        in an iterative way until a certain criterion is met. To avoid infinite loops, we
        add a maximum number of times the while loop is run.

        This rule would allow to reach a minimum step_size of 10^-alpha.

        if
        alpha = numpy.power(10,gamma)
        delta = 3
        step_size = numpy.power(10, -delta)
        with armijo rule we can get to step_size from initial alpha by repeating the while loop k times
        where
        alpha / 2^k = step_size
        10^gamma / 2^k = 10^-delta
        2^k = 10^(gamma+delta)
        k = gamma+delta / log10(2) \\approx 3.3 * (gamma+delta)

        if we would take by default delta = gamma
        kmax = numpy.ceil ( 2 * gamma / numpy.log10(2) )
        kmax = numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))

        '''
        self.alpha_orig=alpha
        
        self.beta=beta
        self.kmax=kmax
        if self.kmax is None:
            self.kmax = numpy.ceil (2 * numpy.log10(alpha) / numpy.log10(2))
                
    def __call__(self, algorithm): 

        """
        """
        k = 0
        self.alpha=self.alpha_orig
        f_x = algorithm.objective_function(algorithm.x)
        
        self.x_update = algorithm.x_update
        
        self.x_armijo = algorithm.x.copy()

        while k < self.kmax:
            # self.x - alpha * self.x_update
            self.x_update.multiply(self.alpha, out=self.x_armijo)
            algorithm.x.subtract(self.x_armijo, out=self.x_armijo)

            f_x_a = algorithm.objective_function(self.x_armijo)
            sqnorm = self.x_update.squared_norm()
            if f_x_a - f_x <= - ( self.alpha/2. ) * sqnorm:
                break
            k += 1.
            self.alpha *= self.beta

        if k == self.kmax:
            raise ValueError('Could not find a proper step_size in {} loops. Consider increasing alpha.'.format(self.kmax))
        return self.alpha
     
