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
            The algorithm object.
        """    
        pass

class Sensitivity(Preconditioner):
    
    """
    Sensitivity preconditioner class. 
    
    In each call to the preconditioner the `algorithm.gradient_update` is multiplied by :math:` y/(A^T \mathbf{1})` where :math:`A` is an operator, :math:`\mathbf{1}` is an object in the range of the operator filled with ones and :math:`y` is an optional reference image.  

    Parameters
    ----------
    operator : CIL Operator 
        The operator used for sensitivity computation.
    reference : DataContainer e.g. ImageData, optional
        The reference data, an object in the domain of the operator
    """
            

    def __init__(self, operator, reference = None): 
        
        super(Sensitivity, self).__init__()
        self.operator = operator
        self.reference = reference
            
        self.compute_preconditioner_matrix()
        
 
    def compute_preconditioner_matrix(self):
        
        """
        Compute the sensitivity. :math:`A^T \mathbf{1}` where :math:`A` is the operator and :math:`\mathbf{1}` is an object in the range of the operator filled with ones.        
        Then perform safe division by the sensitivity to store the preconditioner array :math:` y/(A^T \mathbf{1})`
        """
        self.array=self.operator.adjoint(self.operator.range_geometry().allocate(value=1.0))

        if self.reference is not None:
            self.reference.divide(self.array, where=self.array.as_array()>0 , out=self.array )
        else:            
            self.operator.range_geometry().allocate(value=1.0).divide(self.array, where=self.array.as_array()>0 , out=self.array )
                                        
    def __call__(self, algorithm): 
        
        """
        Update the preconditioner.

        Parameters
        ----------
        algorithm : object
            The algorithm object.
        """
        
        algorithm.gradient_update.multiply(self.array, out=algorithm.gradient_update)

class AdaptiveSensitivity(Sensitivity):

    """
    Adaptive Sensitivity preconditioner class. 
    
    In each call to the preconditioner the `algorithm.gradient_update` is multiplied by :math:` (x+\delta) \odot y /(A^T \mathbf{1})` where :math:`A` is an operator,  :math:`\mathbf{1}` is an object in the range of the operator filled with ones and :math:`y` is an optional reference image.  
    The point :math:`x` is the current iteration and :math:`delta` is a small positive float. 
    

    Parameters
    ----------
    operator : CIL object
        The operator used for sensitivity computation.
    delta : float, optional
        The delta value for the preconditioner.
    iterations : int, optional
        The maximum number of iterations before the preconditoner is frozen and no-longer updates.
    reference : DataContainer e.g. ImageData, optional
        The reference data, an object in the domain of the operator

    Reference
    ---------
    Twyman R, Arridge S, Kereta Z, Jin B, Brusaferri L, Ahn S, Stearns CW, Hutton BF, Burger IA, Kotasidis F, Thielemans K. An Investigation of Stochastic Variance Reduction Algorithms for Relative Difference Penalized 3D PET Image Reconstruction. IEEE Trans Med Imaging. 2023 Jan;42(1):29-41. doi: 10.1109/TMI.2022.3203237. Epub 2022 Dec 29. PMID: 36044488.

    """
    
    def __init__(self, operator, delta = 1e-6, iterations = 100, reference=None): 

        self.iterations = iterations 
        self.delta = delta
        self.freezing_point = operator.domain_geometry().allocate(0)  
        
        super(AdaptiveSensitivity, self).__init__(operator=operator, reference=reference)
    
    def __call__(self, algorithm):
    
        if algorithm.iteration<=self.iterations:
            self.array.multiply(algorithm.x + self.delta, out=self.freezing_point)
            algorithm.gradient_update.multiply(self.freezing_point, out=algorithm.gradient_update)            
        else:  
            algorithm.gradient_update.multiply(self.freezing_point, out=algorithm.gradient_update)

        
        
class AdaGrad(Preconditioner):

    """
    This Adaptive Gradient method multiplies the gradient, :math`\nabla f(x_k)`, by :math:`1/\sqrt{ s_k^2 +\epsilon}`. Where :math:`s_k^2=s^2_{k-1}+diag((\nabla f(x_k))(\nabla f(x_k))^T)``. 
    
    Reference
    ---------
    Duchi, J., Hazan, E. and Singer, Y., 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of machine learning research, 12(7).
    """
    
    
    def __init__(self, epsilon=1e-8):
        self.epsilon=epsilon  
        self.gradient_accumulator=None

    def __call__(self, algorithm):
        """
        Method to __call__ the preconditioner. This multiplies the gradient, :math`\nabla f(x_k)`, by :math:`1/\sqrt{ s_k^2 +\epislon}`. Where :math:`s_k^2=s^2_{k-1}+diag((\nabla f(x_k))(\nabla f(x_k))^T)`. 

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        """    
              
        if self.gradient_accumulator is None:
            self.gradient_accumulator = algorithm.gradient_update.multiply(algorithm.gradient_update)
            
        
        else: 
            self.gradient_accumulator.add(  algorithm.gradient_update.multiply(algorithm.gradient_update), out=self.gradient_accumulator)
            

        algorithm.gradient_update.divide(( self.gradient_accumulator+self.epsilon).sqrt(), out=algorithm.gradient_update)

        
class Adam(Preconditioner):

    """
    This ADAM method combines the adaptive learning rate of AdaGrad with the idea of moomentum. 
    
    ADAM keeps track of the momentum :math:`g_k=\gamma g_{k-1} +(1-\gamma)\nabla f(x_k).
    
    It also updates the scaling factors  :math:`s_k^2=\beta s^2_{k-1}+(1-\beta) diag((\nabla f(x_k))(\nabla x_k)^T)`. 
    
    The returned `new` gradient is :math:`g_k/\sqrt(s_k^2+\epsilon)`

    
    Reference
    ---------
    Kingma, D.P. and Ba, J., 2014. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980.

    """
    
    
    def __init__(self, gamma=0.9, beta =0.999, epsilon=1e-8):
        self.gamma=gamma
        self.beta=beta 
        self.gradient_accumulator=None
        self.scaling_factor_accumulator=None
        self.epsilon=epsilon 

    def __call__(self, algorithm):
        """
        Method to __call__ the preconditioner, updating `self.gradient_update` with the preconditioned gradient.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
        """    
        
        if self.gradient_accumulator is None:
            self.gradient_accumulator = algorithm.gradient_update.copy()
            self.scaling_factor_accumulator=  algorithm.gradient_update.multiply(algorithm.gradient_update)
        
        else: 
            self.gradient_accumulator.sapyb(self.gamma, algorithm.gradient_update, (1-self.gamma), out=self.gradient_accumulator)
            self.scaling_factor_accumulator.sapyb(self.beta,  algorithm.gradient_update.multiply(algorithm.gradient_update), (1-self.beta), out=self.scaling_factor_accumulator)
        
        self.gradient_accumulator.divide(( self.scaling_factor_accumulator+self.epsilon).sqrt(), out=algorithm.gradient_update)
        
        