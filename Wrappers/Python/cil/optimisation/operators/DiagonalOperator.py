#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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

import numpy as np
from cil.framework import ImageData
from cil.optimisation.operators import LinearOperator, BlockOperator
from cil.framework import BlockDataContainer

class DiagonalOperator(LinearOperator):

    r"""DiagonalOperator

    Performs an element-wise multiplication, i.e., `Hadamard Product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#:~:text=In%20mathematics%2C%20the%20Hadamard%20product,elements%20i%2C%20j%20of%20the>`_
    of a :class:`DataContainer` `x` and :class:`DataContainer` `diagonal`, `d` .

    .. math:: (D\circ x) = \sum_{i,j}^{M,N} D_{i,j} x_{i, j}

    In matrix-vector interpretation, if `D` is a :math:`M\times N` dense matrix and is flattened, we have a :math:`M*N \times M*N` vector.
    A sparse diagonal matrix, i.e., :class:`DigaonalOperator` can be created if we add the vector above to the main diagonal.
    If the :class:`DataContainer` `x` is also flattened, we have a :math:`M*N` vector.
    Now, matrix-vector multiplcation is allowed and results to a :math:`(M*N,1)` vector. After reshaping we recover a :math:`M\times N` :class:`DataContainer`.

    Parameters
    ----------
    diagonal : DataContainer
        DataContainer with the same dimensions as the data to be operated on.
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the diagonal geometry directly. default=None .

    """
    def __init__(self, diagonal, domain_geometry=None):
        if domain_geometry is None:
            domain_geometry = diagonal.geometry
        super(DiagonalOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        if isinstance(diagonal, BlockDataContainer):
            self.operator = _BlockDiagonalOperator(diagonal)          
        else:
            self.operator = _DiagonalOperator(diagonal, domain_geometry)
        self.diagonal = diagonal
    def direct(self,x,out=None):
        "Returns :math:`D\circ x` "
        return self.operator.direct(x,out=out)

    def adjoint(self,x, out=None):
        "Returns :math:`D^*\circ x` "
        return self.operator.adjoint(x,out=out)
    
    def calculate_norm(self, **kwargs):
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`

        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}
        """
        return self.operator.calculate_norm(**kwargs)


class _DiagonalOperator(LinearOperator):

    r"""DiagonalOperator

    Performs an element-wise multiplication, i.e., `Hadamard Product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#:~:text=In%20mathematics%2C%20the%20Hadamard%20product,elements%20i%2C%20j%20of%20the>`_
    of a :class:`DataContainer` `x` and :class:`DataContainer` `diagonal`, `d` .

    .. math:: (D\circ x) = \sum_{i,j}^{M,N} D_{i,j} x_{i, j}

    In matrix-vector interpretation, if `D` is a :math:`M\times N` dense matrix and is flattened, we have a :math:`M*N \times M*N` vector.
    A sparse diagonal matrix, i.e., :class:`DigaonalOperator` can be created if we add the vector above to the main diagonal.
    If the :class:`DataContainer` `x` is also flattened, we have a :math:`M*N` vector.
    Now, matrix-vector multiplcation is allowed and results to a :math:`(M*N,1)` vector. After reshaping we recover a :math:`M\times N` :class:`DataContainer`.

    Parameters
    ----------
    diagonal : DataContainer
        DataContainer with the same dimensions as the data to be operated on.
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the diagonal geometry directly. default=None .

    """
    def __init__(self, diagonal, domain_geometry=None):
        if domain_geometry is None:
            domain_geometry = diagonal.geometry.copy()
        super(_DiagonalOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal

    def direct(self,x,out=None):
        "Returns :math:`D\circ x` "
        if out is None:
            return self.diagonal * x
        else:
            self.diagonal.multiply(x,out=out)
        return out

    def adjoint(self,x, out=None):
        "Returns :math:`D^*\circ x` "
        return self.diagonal.conjugate().multiply(x,out=out)

    def calculate_norm(self, **kwargs):
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`

        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}
        """
        return self.diagonal.abs().max()
    
class _BlockDiagonalOperator(LinearOperator):

    r"""BlockDiagonalOperator

    Performs an element-wise multiplication, i.e., `Hadamard Product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#:~:text=In%20mathematics%2C%20the%20Hadamard%20product,elements%20i%2C%20j%20of%20the>`_
    of a :class:`DataContainer` `x` and :class:`DataContainer` `diagonal`, `d` .

    .. math:: (D\circ x) = \sum_{i,j}^{M,N} D_{i,j} x_{i, j}

    In matrix-vector interpretation, if `D` is a :math:`M\times N` dense matrix and is flattened, we have a :math:`M*N \times M*N` vector.
    A sparse diagonal matrix, i.e., :class:`DigaonalOperator` can be created if we add the vector above to the main diagonal.
    If the :class:`DataContainer` `x` is also flattened, we have a :math:`M*N` vector.
    Now, matrix-vector multiplcation is allowed and results to a :math:`(M*N,1)` vector. After reshaping we recover a :math:`M\times N` :class:`DataContainer`.

    Parameters
    ----------
    diagonal : BlockDataContainer
        BlockDataContainer with the same dimensions as the data to be operated on.
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the diagonal geometry directly. default=None .

    """
    def __init__(self, diagonal, domain_geometry=None):
        if domain_geometry is None:
            domain_geometry = diagonal.geometry.copy()
        super(_BlockDiagonalOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal
        self.diagonal_operator_list = [ DiagonalOperator(diagonal[i]) for i in range(len(diagonal)) ]

    def direct(self,x,out=None):
        "Returns :math:`D\circ x` "
        if out is None:
            out = x.copy()
        for i in range(len(self.diagonal)):
            self.diagonal_operator_list[i].direct(x[i], out=out[i])
        return out

    def adjoint(self,x, out=None):
        "Returns :math:`D^*\circ x` "
        if out is None:
            out = x.copy()
        for i in range(len(self.diagonal)):
            self.diagonal_operator_list[i].adjoint(x[i], out=out[i])
        return out

    def calculate_norm(self, **kwargs):
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`

        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}
        """
        norms = [ op.calculate_norm(**kwargs) for op in self.diagonal_operator_list ]
        return max(norms)