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
        pos_ind = sensitivity_np>0
        array_np = np.zeros(self.operator.domain_geometry().allocate().shape)

        if self.reference is not None:
            array_np[pos_ind ] = self.reference.as_array()[pos_ind ]/sensitivity_np[pos_ind ]
        else:            
            array_np[pos_ind ] = (1./sensitivity_np[pos_ind])
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

        
        
class AdaGrad(Preconditioner):

    """
    TODO:

    """
    
    
    def __init__(self, epsilon=1e-4):
        self.epsilon=epsilon  

    def __call__(self, algorithm):
        """
        Method to __call__ the preconditioner. #TODO:

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        """    
              
        
        gradient_np = algorithm.x_update.as_array().copy()
        array_np=np.ones_like(gradient_np)/np.sqrt( np.square(gradient_np)+self.epsilon)
        self.array = algorithm.x.geometry.allocate(0)
        self.array.fill(array_np) 
        
        algorithm.x_update.multiply(self.array, out=algorithm.x_update)
        
        
class Adam(Preconditioner):

    """
    TODO:

    """
    
    
    def __init__(self, gamma=0.9, beta =0.999):
        self.gamma=gamma
        self.beta=beta 
        self.gradient_accumulator=None
        self.scaling_factor_accumulator=None

    def __call__(self, algorithm):
        """
        Method to __call__ the preconditioner. #TODO:

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        """    
        
        if self.gradient_accumulator is None:
            self.gradient_accumulator = algorithm.x_update.copy()
            self.scaling_factor_accumulator=  algorithm.x_update.multiply(algorithm.x_update)
        
        else: 
            self.gradient_accumulator.sapyb(self.gamma, algorithm.x_update, (1-self.gamma), out=self.gradient_accumulator)
            self.scaling_factor_accumulator.sapyb(self.beta,  algorithm.x_update.multiply(algorithm.x_update), (1-self.beta), out=self.scaling_factor_accumulator)
        
        sensitivity_np = np.sqrt(self.scaling_factor_accumulator.as_array())
        pos_ind = sensitivity_np>0
        array_np = np.zeros(algorithm.x.geometry.allocate().shape)

        array_np[pos_ind ] = (1./sensitivity_np[pos_ind])
        self.array = algorithm.x.geometry.allocate(0)
        self.array.fill(array_np) 
        
        self.gradient_accumulator.multiply(self.array, out=algorithm.x_update)