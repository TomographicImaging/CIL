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

    r"""
    Abstract base class for Preconditioner objects. The `apply` method of this class takes an initialised CIL function as an argument and modifies a provided  `gradient`.
   
    
    Methods
    -------
    apply(x)
        Abstract method to call the preconditioner.
    """

    def __init__(self):
        '''Initialises the preconditioner
        '''
        pass

    @abstractmethod
    def apply(self, algorithm, gradient, out=None):
        r"""
        Abstract method to apply the preconditioner.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm object.
    
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, optional, default is None 
            The modified gradient to return 
            
        """
        pass


class Sensitivity(Preconditioner):

    r"""
    Sensitivity preconditioner class. 

    In each call to the preconditioner the `gradient` is multiplied by :math:` y/(A^T \mathbf{1})` where :math:`A` is an operator, :math:`\mathbf{1}` is an object in the range of the operator filled with ones and :math:`y` is an optional reference image.  

    Parameters
    ----------
    operator : CIL Operator 
        The operator used for sensitivity computation.
    reference : DataContainer e.g. ImageData, optional
        The reference data, an object in the domain of the operator. Recommended to be a best guess reconstruction. 
    """

    def __init__(self, operator, reference=None):

        super(Sensitivity, self).__init__()
        self.operator = operator
        self.reference = reference

        self.compute_preconditioner_matrix()

    def compute_preconditioner_matrix(self):
        r"""
        Compute the sensitivity. :math:`A^T \mathbf{1}` where :math:`A` is the operator and :math:`\mathbf{1}` is an object in the range of the operator filled with ones.        
        Then perform safe division by the sensitivity to store the preconditioner array :math:` y/(A^T \mathbf{1})`
        """
        self.array = self.operator.adjoint(
            self.operator.range_geometry().allocate(value=1.0))

        if self.reference is not None:
            self.reference.divide(
                self.array, where=self.array.as_array() > 0, out=self.array)
        else:
            self.operator.range_geometry().allocate(value=1.0).divide(
                self.array, where=self.array.as_array() > 0, out=self.array)

    def apply(self, algorithm, gradient, out=None):
        r"""
        Update the preconditioner.

        Parameters
        ----------
        algorithm : object
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, optional, default is None 
            The modified gradient to return 
        """
        if out is None:
            out = gradient.copy()
        gradient.multiply(
            self.array, out=out)
        return out


class AdaptiveSensitivity(Sensitivity):

    r"""
    Adaptive Sensitivity preconditioner class. 

    In each call to the preconditioner the `gradient` is multiplied by :math:` (x+\delta) \odot y /(A^T \mathbf{1})` where :math:`A` is an operator,  :math:`\mathbf{1}` is an object in the range of the operator filled with ones and :math:`y` is an optional reference image.  
    The point :math:`x` is the current iteration and :math:`delta` is a small positive float. 


    Parameters
    ----------
    operator : CIL object
        The operator used for sensitivity computation.
    delta : float, optional
        The delta value for the preconditioner.
    iterations : int, optional
        The maximum number of iterations before the preconditoner is frozen and no-longer updates. Freezing iterations saves computational effort. 
    reference : DataContainer e.g. ImageData, optional
        The reference data, an object in the domain of the operator. Recommended to be a best guess reconstruction. 

    Reference
    ---------
    Twyman R, Arridge S, Kereta Z, Jin B, Brusaferri L, Ahn S, Stearns CW, Hutton BF, Burger IA, Kotasidis F, Thielemans K. An Investigation of Stochastic Variance Reduction Algorithms for Relative Difference Penalized 3D PET Image Reconstruction. IEEE Trans Med Imaging. 2023 Jan;42(1):29-41. doi: 10.1109/TMI.2022.3203237. Epub 2022 Dec 29. PMID: 36044488.

    """

    def __init__(self, operator, delta=1e-6, iterations=100, reference=None):

        self.iterations = iterations
        self.delta = delta
        self.freezing_point = operator.domain_geometry().allocate(0)

        super(AdaptiveSensitivity, self).__init__(
            operator=operator, reference=reference)

    def apply(self, algorithm, gradient, out=None):
        r"""
        Update the preconditioner.

        Parameters
        ----------
        algorithm : object
            The algorithm object.
        gradient : DataContainer
            The calculated gradient to modify
        out : DataContainer, optional, default is None 
            The modified gradient to return 
        """
        if out is None:
            out = gradient.copy()
            
        if algorithm.iteration <= self.iterations:
            self.array.multiply(algorithm.get_output() + self.delta,
                                out=self.freezing_point)
            gradient.multiply(
                self.freezing_point, out=out)
        else:
            gradient.multiply(
                self.freezing_point, out=out)
        
        return out

