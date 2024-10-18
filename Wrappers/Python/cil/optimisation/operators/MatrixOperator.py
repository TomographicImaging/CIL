#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
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
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy
from scipy.sparse.linalg import svds
from cil.framework import VectorGeometry
from cil.optimisation.operators import LinearOperator

class MatrixOperator(LinearOperator):
    r""" Matrix wrapped in a CIL Operator to be used in optimisation algorithms. 

    Parameters      
    ----------
    A: a numpy matrix
        The matrix to be wrapped into a CIL Operator

    """

    def __init__(self,A):
        """Constructor"""
        self.A = A
        M_A, N_A = self.A.shape
        domain_geometry = VectorGeometry(N_A, dtype=A.dtype)
        range_geometry = VectorGeometry(M_A, dtype=A.dtype)
        super(MatrixOperator, self).__init__(domain_geometry=domain_geometry,
                                                   range_geometry=range_geometry)

    def direct(self,x, out=None):
        r"""Returns :math:`Ax`
        
        Parameters
        ----------
        x : DataContainer
            Input data
        out : DataContainer, optional
            Output data, default is None
        Returns
        -------
        DataContainer
            :math:`Ax`
        """
        
        if out is None:
            tmp = self.range_geometry().allocate()
            tmp.fill(numpy.dot(self.A,x.as_array()))
            return tmp
        else:
            # Below use of out is not working, see
            # https://docs.scipy.org/doc/numpy/reference/generated/numpy.dot.html
            # numpy.dot(self.A, x.as_array(), out = out.as_array())
            out.fill(numpy.dot(self.A, x.as_array()))
            return out

    def adjoint(self,x, out=None):
        r"""Returns :math:`A^{T}x`
        
        Parameters
        ----------
        x : DataContainer
            Input data
        out : DataContainer, optional
            Output data, default is None
            
        Returns
        -------
        DataContainer
            :math:`A^{T}x`
        """
        
        if out is None:
            tmp = self.domain_geometry().allocate()
            tmp.fill(numpy.dot(self.A.transpose().conjugate(),x.as_array()))
            return tmp
        else:
            out.fill(numpy.dot(self.A.transpose().conjugate(),x.as_array()))
            return out

    def size(self):
        r"""Returns the size of the matrix
        """
        return self.A.shape
