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
from cil.framework import BlockDataContainer, BlockGeometry
from cil.optimisation.operators import LinearOperator

class DiagonalOperator(LinearOperator):

    r"""DiagonalOperator

    Performs an element-wise multiplication, i.e., `Hadamard Product <https://en.wikipedia.org/wiki/Hadamard_product_(matrices)#:~:text=In%20mathematics%2C%20the%20Hadamard%20product,elements%20i%2C%20j%20of%20the>`_
    of a :class:`DataContainer` `x` and :class:`DataContainer` `diagonal`, `d` .

    .. math:: (D\circ x) = \sum_{i,j}^{M,N} D_{i,j} x_{i, j}

    In matrix-vector interpretation, if `D` is a :math:`M\times N` dense matrix and is flattened, we have a :math:`M*N \times M*N` vector.
    A sparse diagonal matrix, i.e., :class:`DigaonalOperator` can be created if we add the vector above to the main diagonal.
    If the :class:`DataContainer` `x` is also flattened, we have a :math:`M*N` vector.
    Now, matrix-vector multiplcation is allowed and results to a :math:`(M*N,1)` vector. After reshaping we recover a :math:`M\times N` :class:`DataContainer`.

    A :class:`BlockDataContainer` `diagonal` applies one diagonal per block --
    equivalent to a :class:`BlockOperator` with these diagonals on its main
    diagonal and ``ZeroOperator`` s elsewhere, without forming the off-diagonal
    blocks. Blocks may themselves be blocks, to any depth.

    Parameters
    ----------
    diagonal : DataContainer or BlockDataContainer
        DataContainer with the same dimensions as the data to be operated on.
    domain_geometry : ImageGeometry
        Specifies the geometry of the operator domain. If 'None' will use the diagonal geometry directly. default=None .

    """
    def __init__(self, diagonal, domain_geometry=None):
        if isinstance(diagonal, BlockDataContainer):
            self.diagonal_operator_list = [ DiagonalOperator(diagonal[i]) for i in range(len(diagonal)) ]
            if domain_geometry is None:
                # `diagonal.geometry` is None for a block of blocks, because
                # it builds on `el.geometry.copy()` and BlockGeometry has no
                # copy(). Each child operator knows its own domain at whatever
                # depth, so the geometry is composed from them.
                domain_geometry = BlockGeometry(
                    *[op.domain_geometry() for op in self.diagonal_operator_list])
        else:
            self.diagonal_operator_list = None
            if domain_geometry is None:
                domain_geometry = diagonal.geometry.copy()
        super(DiagonalOperator, self).__init__(domain_geometry=domain_geometry,
                                    range_geometry=domain_geometry)
        self.diagonal = diagonal
        # Decided once: a real diagonal is self-adjoint, and conjugate()
        # allocates a whole container per call, which adjoint() cannot afford
        # -- the solvers call it once per iteration. dtype cannot change in
        # place, so mutating the diagonal in place cannot stale this.
        self._is_complex = (self.diagonal_operator_list is None
                            and np.issubdtype(diagonal.dtype,
                                              np.complexfloating))

    def direct(self,x,out=None):
        "Returns :math:`D\circ x` "
        if self.diagonal_operator_list is not None:
            if out is None:
                out = x.copy()
            for i, operator in enumerate(self.diagonal_operator_list):
                operator.direct(x[i], out=out[i])
            return out
        if out is None:
            return self.diagonal * x
        else:
            self.diagonal.multiply(x,out=out)
        return out

    def adjoint(self,x, out=None):
        "Returns :math:`D^*\circ x`, which is :math:`D\circ x` for a real `diagonal` "
        if self.diagonal_operator_list is not None:
            if out is None:
                out = x.copy()
            for i, operator in enumerate(self.diagonal_operator_list):
                operator.adjoint(x[i], out=out[i])
            return out
        if self._is_complex:
            return self.diagonal.conjugate().multiply(x,out=out)
        return self.direct(x, out=out)

    def calculate_norm(self, **kwargs):
        r""" Returns the operator norm of DiagonalOperator which is the :math:`\infty` norm of `diagonal`

        .. math:: \|D\|_{\infty} = \max_{i}\{|D_{i}|\}
        """
        if self.diagonal_operator_list is not None:
            # Block-diagonal, so the largest of the per-block norms.
            return max(operator.calculate_norm(**kwargs)
                       for operator in self.diagonal_operator_list)
        return self.diagonal.abs().max()
