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
# CIL Developers, listed at:
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

from cil.optimisation.operators import (
    LinearOperator,
    IdentityOperator,
    DiagonalOperator,
)


class WeightedOperator(LinearOperator):
    r"""
    Composes an operator :math:`L` with a mutable diagonal weight operator :math:`W` 
    acting on its range, representing :math:`WL`.

    The weights are allocated at construction, filled with ones, and are
    rewritten in place through :attr:`weights` or :meth:`set_weights`. For a
    block-valued :math:`L`, such as a gradient operator, the weights are
    block-valued too.

    Parameters
    ----------
    domain_geometry: cil geometry
        ``Domain(L)``.
    operator: `LinearOperator`, default None
        :math:`L`. ``None`` means the identity, and the stage is skipped.
    tmp_range: DataContainer, default None
        Scratch in ``Range(L)`` to borrow instead of allocating.
    """

    def __init__(self, domain_geometry, operator=None, tmp_range=None):
        # An absent operator and an explicit IdentityOperator both skip the stage.
        self.operator_is_identity = (operator is None
                                     or isinstance(operator,
                                                   IdentityOperator))
        if operator is None:
            operator = IdentityOperator(domain_geometry)

        self.operator = operator

        super(WeightedOperator, self).__init__(
            domain_geometry=domain_geometry,
            range_geometry=operator.range_geometry(),
        )

        range_geometry = operator.range_geometry()
        self.weight_operator = DiagonalOperator(
            range_geometry.allocate(1.0),
            domain_geometry=range_geometry,
        )

        # Only a genuine two-stage composition needs an intermediate.
        if tmp_range is None and not self.operator_is_identity:
            tmp_range = range_geometry.allocate(0)
        self.tmp_range = tmp_range

    @property
    def weights(self):
        """The mutable diagonal weights container in ``Range(L)``. Modify in
        place; rebinding the returned object has no effect."""
        return self.weight_operator.diagonal

    @staticmethod
    def _fill_in_place(target, values):
        """Recurses to the leaves before filling: ``BlockDataContainer.fill``
        is not recursive, and ``Range(L)`` can be a block of blocks."""
        if not hasattr(target, "containers"):
            target.fill(values)
            return
        source_is_block = hasattr(values, "containers")
        for i in range(len(target)):
            WeightedOperator._fill_in_place(
                target[i], values[i] if source_is_block else values)

    def set_weights(self, values):
        """Copies ``values`` into the weights in place, block-aware.

        Parameters
        ----------
        values: scalar, DataContainer or BlockDataContainer
            A container matching the shape of ``Range(L)``, or anything
            broadcastable to it.

        Returns
        -------
        The updated weights container.
        """
        target = self.weights
        self._fill_in_place(target, values)
        return target

    def direct(self, x, out=None):
        r"""Returns :math:`W L x`.

        Parameters
        ----------
        x: DataContainer
            Element in the domain of the operator.
        out: DataContainer or BlockDataContainer, default None
            If out is not None the result is filled in out, otherwise a new
            object is returned.

        Returns
        -------
        DataContainer or BlockDataContainer containing the result.
        """
        if out is None:
            out = self.range_geometry().allocate(0)

        self.operator.direct(x, out=out)
        self.weight_operator.direct(out, out=out)

        return out

    def adjoint(self, x, out=None):
        r"""Returns :math:`L^* W^* x`.

        Parameters
        ----------
        x: DataContainer or BlockDataContainer
            Element in the range of the operator.
        out: DataContainer, default None
            If out is not None the result is filled in out, otherwise a new
            object is returned.

        Returns
        -------
        DataContainer containing the result.
        """
        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.operator_is_identity:
            # Domain(L) is Range(L); L^* is a no-op.
            return self.weight_operator.adjoint(x, out=out)

        self.weight_operator.adjoint(x, out=self.tmp_range)
        return self.operator.adjoint(self.tmp_range, out=out)

    def inverse(self, x, out=None):
        r"""Returns :math:`y` such that :math:`W L y = x`, as :math:`L^* W^{-1} x`.
        Requires an orthogonal, square :math:`L` and strictly positive weights.

        Parameters
        ----------
        x: DataContainer or BlockDataContainer
            Element in the range of the operator.
        out: DataContainer, default None
            If out is not None the result is filled in out, otherwise a new
            object is returned.

        Returns
        -------
        DataContainer containing the result.
        """
        self._require_invertible(self.operator)

        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.operator_is_identity:
            # L^{-1} is a no-op, so divide straight into out.
            x.divide(self.weights, out=out)
            return out

        x.divide(self.weights, out=self.tmp_range)
        return self.operator.adjoint(self.tmp_range, out=out)

    def inverse_adjoint(self, x, out=None):
        r"""Returns :math:`y` such that :math:`(W L)^* y = x`, as :math:`W^{-1} L x`.
        Requires an orthogonal, square :math:`L` and strictly positive weights.

        Parameters
        ----------
        x: DataContainer
            Element in the domain of the operator.
        out: DataContainer or BlockDataContainer, default None
            If out is not None the result is filled in out, otherwise a new
            object is returned.

        Returns
        -------
        DataContainer or BlockDataContainer containing the result.
        """
        self._require_invertible(self.operator)

        if out is None:
            out = self.range_geometry().allocate(0)

        self.operator.direct(x, out=out)
        out.divide(self.weights, out=out)

        return out
    
    @staticmethod
    def _require_invertible(operator):
        r"""Raises unless the operator's inverse exists and equals its
        adjoint: orthogonal *and* square. Orthogonality alone is not enough:
        a padding ``WaveletOperator`` is a tall isometry (:math:`L^*L = I`
        but :math:`LL^* \neq I`)."""
        if (operator.is_orthogonal()
                and operator.domain_geometry().shape
                == operator.range_geometry().shape):
            return
        raise ValueError(
            "Forming L^-1 requires an orthogonal, square L, so that "
            "L^-1 = L^*, and {} is not. If using IRLS use form='block', "
            "which never forms an inverse.".format(type(operator).__name__))

    def is_invertible(self):
        r"""Returns True when :meth:`inverse` and :meth:`inverse_adjoint` can
        be used: :math:`L` is orthogonal and square."""
        try:
            self._require_invertible(self.operator)
        except ValueError:
            return False
        return True
