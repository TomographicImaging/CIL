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
import numpy as np
class Preconditioner(ABC):

    """
    Abstract base class for Preconditioner objects.

    Parameters
    ----------
    array : numpy.ndarray, optional
        The preconditioner array.

    Methods
    -------
    __call__(x)
        Abstract method to call the preconditioner.
    """
    
    
    def __init__(self):
        pass   

    @abstractmethod
    def __call__(self, algorithm):
        """
        Abstract method to __call__ the preconditioner.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithmrithm object.
        """    
        pass

class Sensitivity(Preconditioner):
    
    """
    Sensitivity preconditioner class. TODO: need a reference for this! 

    Parameters
    ----------
    operator : object
        The operator used for sensitivity computation.
    reference : object, optional
        The reference data.
    """
            

    def __init__(self, operator, reference = None): 
        
        super(Sensitivity, self).__init__()
        self.operator = operator
        self.reference = reference
            
        self.compute_preconditioner_matrix()
        
    def compute_sensitivity(self):
        
        """
        Compute the sensitivity. :math:` A^T \mathbf{1}` where :math:`A` is the operator and :math:`\mathbf{1}` is an object in the range of the operator filled with ones. 
        """        
        
        self.sensitivity = self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))
 
    def compute_preconditioner_matrix(self):
        
        """
        Perform safe division by the sensitivity.
        """
        self.compute_sensitivity()
        sensitivity_np = self.sensitivity.as_array()
        self.pos_ind = sensitivity_np>0
        array_np = np.zeros(self.operator.domain_geometry().allocate().shape)

        if self.reference is not None:
            array_np[self.pos_ind ] = self.reference.as_array()[self.pos_ind ]/sensitivity_np[self.pos_ind ]
        else:            
            array_np[self.pos_ind ] = (1./sensitivity_np[self.pos_ind])
        self.array = self.operator.domain_geometry().allocate(0)
        self.array.fill(array_np) 
                                        
    def __call__(self, algorithm): 
        
        """
        Update the preconditioner.

        Parameters
        ----------
        algorithm : object
            The algorithm object.
        """
        
        algorithm.x_update.multiply(self.array, out=algorithm.x_update)

class AdaptiveSensitivity(Sensitivity):

    """
    Adaptive Sensitivity preconditioner class. TODO: need a reference for this 

    Parameters
    ----------
    operator : object
        The operator used for sensitivity computation.
    delta : float, optional
        The delta value for the __call__.
    iterations : int, optional
        The maximum number of iterations.
    array : numpy.ndarray, optional
        The preconditioner array.

    """
    
    def __init__(self, operator, delta = 1e-6, iterations = 100, reference=None): 

        self.iterations = iterations 
        self.delta = delta
        self.freezing_point = operator.domain_geometry().allocate(0)  
        
        super(AdaptiveSensitivity, self).__init__(operator=operator, reference=reference)
    
    def __call__(self, algorithm):
    
        if algorithm.iteration<=self.iterations:
            self.array.multiply(algorithm.x + self.delta, out=self.freezing_point)
            algorithm.x_update.multiply(self.freezing_point, out=algorithm.x_update)            
        else:  
            algorithm.x_update.multiply(self.freezing_point, out=algorithm.x_update)

        