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

class MomentumCoefficient(ABC):
    '''Abstract base class for MomentumCoefficient objects. The `__call__` method of this class returns the momentum coefficient for the given iteration.
    '''
    def __init__(self):
        '''Initialises the meomentum coefficient object.
        '''
        pass
    
    @abstractmethod
    def __call__(self, algorithm):
        '''Returns the momentum coefficient for the given iteration.
        
        Parameters
        ----------
        algorithm: CIL Algorithm
            The algorithm object.
        '''
        
        pass
    
class ConstantMomentum(MomentumCoefficient):
    
    '''MomentumCoefficient object that returns a constant momentum coefficient.
    
    Parameters
    ----------
    momentum: float
        The momentum coefficient.
    '''
    
    def __init__(self, momentum):
        self.momentum = momentum
        
    def __call__(self, algorithm):
        return self.momentum
    
class NesterovMomentum(MomentumCoefficient):
    
    '''MomentumCoefficient object that returns the Nesterov momentum coefficient.
    
    Parameters
    ----------
    t: float
        The initial value for the momentum coefficient.
    '''
    
    def __init__(self, t= 1):
        self.t = 1
        
    def __call__(self, algorithm):
        self.t_old = self.t
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        return (self.t_old-1)/self.t
        