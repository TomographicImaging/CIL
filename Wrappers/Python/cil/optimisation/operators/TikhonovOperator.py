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

import logging

from cil.optimisation.operators import (
    LinearOperator,
    IdentityOperator,
    BlockOperator,
    ScaledOperator,
    WeightedOperator,
)

log = logging.getLogger(__name__)

VALID_FORMS = ('block', 'standard')


def _build_tikhonov_operator(operator, struct_operator=None, regalpha=0.0,
                             form='block', solution_geometry=None):
    r"""Builds the operator :math:`K` a Krylov solver iterates on for the generalised Tikhonov problem :math:`\min_u \|Au - b\|_2^2 + \alpha^2 \|Lu\|_2^2`.
    Private: LSQR and CGLS call this from ``set_up``.

    Parameters
    ----------
    operator: `LinearOperator`
        The forward model :math:`A`.
    struct_operator: `LinearOperator`, default None
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha: float, default 0.0
        The regularisation parameter :math:`\alpha`. Zero returns the bare
        ``operator`` unwrapped.
    form: {'block', 'standard'}, default 'block'
        'block' builds :math:`K = [A;\ \alpha L]` (`BlockTikhonovOperator`),
        which accepts any :math:`L` and warm starts; 'standard' builds
        :math:`K = A L^{-1}` (`TikhonovOperator`), which is cheaper but needs
        an orthogonal, square :math:`L`, and LSQR cannot warm start in it.
    solution_geometry: cil geometry, default None
        The physical solution space. Defaults to ``operator.domain_geometry()``.

    Returns
    -------
    `LinearOperator` -- the class records which form was built.
    """
    if form not in VALID_FORMS:
        raise ValueError(
            "form must be one of {}, got {!r}.".format(VALID_FORMS, form))

    if regalpha < 0:
        raise ValueError(
            "regalpha must be non-negative, got {}.".format(regalpha))

    if solution_geometry is None:
        solution_geometry = operator.domain_geometry()

    if regalpha == 0:
        return operator

    if form == 'standard':
        return TikhonovOperator(
            operator,
            solution_geometry,
            struct_operator=struct_operator,
            regalpha=regalpha,
        )

    return BlockTikhonovOperator(
        operator,
        solution_geometry,
        struct_operator=struct_operator,
        regalpha=regalpha,
    )


class BlockTikhonovOperator(BlockOperator):
    r"""Stacked Tikhonov operator :math:`K = [A;\ \alpha L]`, regularising directly in the physical solution space.
    It never forms :math:`L^{-1}`, so it is the only valid choice when
    :math:`L` is not invertible.

    Parameters
    ----------
    operator: `LinearOperator`
        The forward model :math:`A`.
    solution_geometry: cil geometry
        The physical solution space, ``Domain(A)``.
    struct_operator: `LinearOperator`, default None
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha: float, default 0.0
        The regularisation parameter :math:`\alpha`.
    tmp_domain: DataContainer, default None
        Accumulator the allocation-free adjoint needs when :math:`L \neq I`;
        pass one to borrow it instead of allocating.
    """

    def __init__(self, operator, solution_geometry, struct_operator=None,
                 regalpha=0.0, tmp_domain=None):
        self.operator = operator

        self.struct_is_identity = (struct_operator is None
                                   or isinstance(struct_operator,
                                                 IdentityOperator))
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)
        self.struct_operator = struct_operator
        self.scaled_reg_operator = ScaledOperator(struct_operator, regalpha)

        # Accumulator for the allocation-free adjoint; needed whenever L != I.
        if tmp_domain is None and not self.struct_is_identity:
            tmp_domain = solution_geometry.allocate(0)
        self.tmp_domain = tmp_domain

        super(BlockTikhonovOperator, self).__init__(
            self.operator,
            self.scaled_reg_operator,
        )

    @property
    def regalpha(self):
        """The current Tikhonov regularisation parameter."""
        return self.scaled_reg_operator.scalar

    @regalpha.setter
    def regalpha(self, value):
        """Update the regularisation parameter in place."""
        self.scaled_reg_operator.scalar = value

    def adjoint(self, x, out=None):
        r"""Returns :math:`K^* x = A^* x_0 + \alpha L^* x_1`.
        Overrides ``BlockOperator.adjoint``, which allocates a domain
        container on every call; this writes through ``out`` and the
        ``tmp_domain`` accumulator instead. Only the two-row shape is handled
        here; anything else falls back to the base class.

        Parameters
        ----------
        x: BlockDataContainer
            Element in the range of the operator.
        out: DataContainer, default None
            If out is not None the result is filled in out, otherwise a new
            object is returned.

        Returns
        -------
        DataContainer containing the result.
        """
        if self.shape != (2, 1):
            return super(BlockTikhonovOperator, self).adjoint(x, out=out)

        if out is None:
            out = self.domain_geometry().allocate(0)

        self.operator.adjoint(x[0], out=out)

        if self.tmp_domain is None:
            # L = I, so the second row is a plain scaling and needs no
            # accumulator at all.
            return out.sapyb(1.0, x[1], self.regalpha, out=out)

        self.struct_operator.adjoint(x[1], out=self.tmp_domain)
        return out.sapyb(1.0, self.tmp_domain, self.regalpha, out=out)


class TikhonovOperator(LinearOperator):
    r"""Standard-form Tikhonov operator :math:`K = A L^{-1}`, iterating in ``Range(L)``; the physical solution is :math:`u = L^{-1} x`.
    :math:`\alpha` is not applied here -- in standard form it enters the
    solver's own recurrence -- and is stored only so callers can check the
    two agree. A plain :math:`L` must be orthogonal and square, so that
    :math:`L^{-1} = L^*`; a `WeightedOperator` is inverted through its own
    ``inverse`` instead. LSQR cannot warm start in this form: the penalty
    lands on the step, not the solution.

    Parameters
    ----------
    operator: `LinearOperator`
        The forward model :math:`A`.
    solution_geometry: cil geometry
        The physical solution space, ``Domain(L)``.
    struct_operator: `LinearOperator`, default None
        The structural operator :math:`L`. ``None`` means the identity.
    regalpha: float, default 0.0
        The regularisation parameter :math:`\alpha`, stored for bookkeeping.
    tmp_domain: DataContainer, default None
        Staging buffer needed when :math:`L \neq I`; pass one to borrow it
        instead of allocating.
    """

    def __init__(self, operator, solution_geometry, struct_operator=None,
                 regalpha=0.0, tmp_domain=None):
        self.operator = operator
        self.regalpha = regalpha

        self.struct_is_identity = (struct_operator is None
                                   or isinstance(struct_operator,
                                                 IdentityOperator))
        if struct_operator is None:
            struct_operator = IdentityOperator(solution_geometry)
        self.struct_operator = struct_operator

        # A WeightedOperator carries a true inverse pair and guards it
        # itself; any other L is inverted as L^*, guarded in _inverse. Either
        # way the check fires where the inverse is formed, not here.
        self._weighted_operator = (struct_operator
                                   if isinstance(struct_operator,
                                                 WeightedOperator) else None)

        # With L = I the transformation is K = A exactly: no staging buffer,
        # and the degenerate path costs the same as an unregularised solve.
        if tmp_domain is None and not self.struct_is_identity:
            tmp_domain = solution_geometry.allocate(0)
        self.tmp_domain = tmp_domain

        super(TikhonovOperator, self).__init__(
            domain_geometry=struct_operator.range_geometry(),
            range_geometry=operator.range_geometry(),
        )

    def _inverse(self, x, out=None):
        r"""Apply :math:`L^{-1}`: the weighted wrapper's true inverse, or
        :math:`L^*` for a plain L, which must be orthogonal and square."""
        if self._weighted_operator is not None:
            return self._weighted_operator.inverse(x, out=out)
        WeightedOperator._require_invertible(self.struct_operator)
        return self.struct_operator.adjoint(x, out=out)

    def _inverse_adjoint(self, x, out=None):
        r"""Apply :math:`L^{-*}`: the weighted wrapper's true inverse adjoint,
        or :math:`L` itself, since :math:`(L^{-1})^* = (L^*)^* = L` for an
        orthogonal square L."""
        if self._weighted_operator is not None:
            return self._weighted_operator.inverse_adjoint(x, out=out)
        WeightedOperator._require_invertible(self.struct_operator)
        return self.struct_operator.direct(x, out=out)

    def solution_geometry(self):
        """The physical solution space, Domain(L)."""
        return self.struct_operator.domain_geometry()

    def map_from_solution(self, u, out=None):
        r"""Map a physical solution into the iterate space: :math:`x = L u`."""
        return self.struct_operator.direct(u, out=out)

    def map_to_solution(self, x, out=None):
        r"""Map an iterate back to the physical solution: :math:`u = L^{-1} x`."""
        return self._inverse(x, out=out)

    def direct(self, x, out=None):
        r"""Apply :math:`K x = A L^{-1} x`."""
        if out is None:
            out = self.range_geometry().allocate(0)

        if self.tmp_domain is None:
            # L = I, so K = A.
            return self.operator.direct(x, out=out)

        self._inverse(x, out=self.tmp_domain)
        return self.operator.direct(self.tmp_domain, out=out)

    def adjoint(self, x, out=None):
        r"""Apply :math:`K^* x = L^{-*} A^* x`."""
        if out is None:
            out = self.domain_geometry().allocate(0)

        if self.tmp_domain is None:
            return self.operator.adjoint(x, out=out)

        self.operator.adjoint(x, out=self.tmp_domain)
        return self._inverse_adjoint(self.tmp_domain, out=out)
