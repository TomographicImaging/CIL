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

from cil.framework import BlockDataContainer
from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators.TikhonovOperator import (
    BlockTikhonovOperator,
    TikhonovOperator,
    _build_tikhonov_operator,
)
import numpy
import logging
import warnings

log = logging.getLogger(__name__)


class CGLS(Algorithm):

    r'''Conjugate Gradient Least Squares (CGLS) algorithm

    The Conjugate Gradient Least Squares (CGLS) algorithm is commonly used for solving large systems of linear equations, due to its fast convergence.

    Problem:

    .. math::

      \min_x || A x - b ||^2_2

    Optionally, with regularisation:

    1. Standard Tikhonov regularisation

    .. math::

        \min_x \|Ax - b\|_2^2 + \alpha^2 \|x\|_2^2

    2. Structured Tikhonov regularisation

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_2^2

    For structured regularisation, the operator :math:`L` can be provided as `struct_operator`. If not provided, it defaults to the identity operator.

    The regularised problem can be iterated on in two forms, chosen with `form`:

    * ``form='block'`` (the default) iterates on the stacked operator
      :math:`K = [A;\ \alpha L]` against :math:`[b;\ 0]`. It accepts any
      :math:`L`, at the price of range containers spanning both
      :math:`\mathrm{Range}(A)` and :math:`\mathrm{Range}(L)`.

    * ``form='standard'`` changes variables to :math:`z = Lx` and iterates on
      :math:`K = A L^{-1}`, with the :math:`\alpha^2 x` term subtracted
      explicitly in the recurrence. It is cheaper -- no stacked containers --
      but requires an orthogonal, square :math:`L`.


    Parameters
    ------------
    operator : Operator
        Linear operator for the inverse problem
    initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
        Initial guess 
    data : DataContainer in the range of the operator 
        Acquired data to reconstruct
    alpha : float, optional
        Regularisation parameter. Default is 0 (no regularisation).
    struct_operator : Operator, optional
        Structured operator for regularisation.
    form : {'block', 'standard'}, default 'block'
        Which formulation of the regularised problem to iterate on, see above.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/cgls/
    '''

    def __init__(self, initial=None, operator=None, data=None, alpha=0, struct_operator=None,
                 form='block', **kwargs):
        '''initialisation of the algorithm
        '''
        super(CGLS, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        self.regalpha = alpha

        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data,
                        struct_operator=struct_operator, form=form)

    def set_up(self, initial, operator, data, struct_operator=None,
               form='block'):
        r'''Initialisation of the algorithm

        Allocates the entire workspace; nothing allocates afterwards.

        Parameters
        ------------
        operator : Operator
            Linear operator for the inverse problem
        initial : (optional) DataContainer in the domain of the operator, default is a DataContainer filled with zeros. 
            Initial guess 
        data : DataContainer in the range of the operator 
            Acquired data to reconstruct
        struct_operator : Operator, optional
            Structured operator for regularisation.
        form : {'block', 'standard'}, default 'block'
            Which formulation of the regularised problem to iterate on. See
            the class documentation.

        '''

        log.info("%s setting up", self.__class__.__name__)

        self.operator = _build_tikhonov_operator(
            operator, struct_operator=struct_operator,
            regalpha=self.regalpha, form=form)

        # The operator's class records which form was built; alpha == 0
        # leaves the bare operator and both flags False.
        self.standard_form = isinstance(self.operator, TikhonovOperator)
        self.block_form = isinstance(self.operator, BlockTikhonovOperator)

        self.data = data

        # Build the workspace by copying initial and data rather than
        # allocating from a geometry, which SIRF data does not have. The
        # values are overwritten in initialise_variables.
        self.initial = initial
        if self.standard_form:
            # Standard form iterates in Range(L): x = L u
            self.x = self.operator.map_from_solution(initial)
        else:
            self.x = initial.copy()
        self.s = self.x.copy()
        self.p = self.x.copy()

        if self.block_form:
            self.r = BlockDataContainer(
                data.copy(), self.operator.struct_operator.direct(initial))
        else:
            self.r = data.copy()
        self.q = self.r.copy()

        self.initialise_variables()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @property
    def supports_warm_start(self):
        """Always True: CGLS subtracts the :math:`\\alpha^2 x` term explicitly,
        so it warm starts in either form."""
        return True

    def solution_geometry(self):
        """The geometry of the physical solution space."""
        if self.standard_form:
            return self.operator.solution_geometry()
        return self.operator.domain_geometry()

    def initialise_variables(self):
        r'''Initialise the variables of the algorithm. Allocates nothing.'''
        # Catch self.regalpha and the operator's regalpha disagreeing.
        operator_alpha = getattr(self.operator, 'regalpha', None)
        if operator_alpha is not None and operator_alpha != self.regalpha:
            raise ValueError(
                "self.regalpha ({}) and self.operator.regalpha ({}) disagree. "
                "Change both together, or re-run set_up.".format(
                    self.regalpha, operator_alpha))

        # Map initial guess to the solution space
        if self.standard_form:
            # Standard form operates in structure space: x_0 = L * u_0
            self.operator.map_from_solution(self.initial, out=self.x)
        else:
            # Block form operates in the physical domain: x_0 = u_0
            self.x.fill(self.initial)

        # Calculate initial residual r = data - operator * x
        self.operator.direct(self.x, out=self.r)
        if self.block_form:
            # Block 0: r[0] = data - operator[0] * x
            self.data.sapyb(1.0, self.r[0], -1.0, out=self.r[0])
            # Block 1: r[1] = 0 - operator[1] * x
            self.r[1].multiply(-1.0, out=self.r[1])
        else:
            # Standard single container form: r = data - operator * x
            self.data.sapyb(1.0, self.r, -1.0, out=self.r)

        # Normal equations residual: s = operator^* r
        self.operator.adjoint(self.r, out=self.s)

        # Add penalty s = s - alpha^2 x for standard Tikhonov
        if self.standard_form:
            self.s.sapyb(1.0, self.x, -(self.regalpha**2), out=self.s)

        self.p.fill(self.s)

        self.norms0 = self.s.norm()
        self.norms = self.norms0

        self.gamma = self.norms0**2

    def update(self):
        '''single iteration'''

        self.operator.direct(self.p, out=self.q)
        delta = self.q.squared_norm()

        # Regularisation term in the normal equations for standard Tikhonov
        if self.standard_form:
            delta += (self.regalpha**2) * self.p.squared_norm()

        alpha = self.gamma/delta

        self.x.sapyb(1, self.p, alpha, out=self.x)
        # self.x += alpha * self.p
        self.r.sapyb(1, self.q, -alpha, out=self.r)
        # self.r -= alpha * self.q

        self.operator.adjoint(self.r, out=self.s)

        # Regularisation term in the normal equations for standard Tikhonov
        if self.standard_form:
            self.s.sapyb(1.0, self.x, -(self.regalpha**2), out=self.s)

        self.norms = self.s.norm()
        self.gamma1 = self.gamma
        self.gamma = self.norms**2
        self.beta = self.gamma/self.gamma1
        # self.p = self.s + self.beta * self.p
        self.p.sapyb(self.beta, self.s, 1, out=self.p)

        if self.norms == 0:
            log.info(
                'The norm of the backprojected residual is zero and so the algorithm is terminated')
            raise StopIteration

    def update_objective(self):
        a = self.r.squared_norm()
        if a is numpy.nan:
            log.info(
                'The objective value is NaN and so the algorithm is terminated')
            raise StopIteration()
        self.loss.append(a)

    def get_output(self, out=None):
        """
        Get the current solution estimate, in the physical solution space.

        Parameters
        ----------
        out : DataContainer, optional
            Buffer to write the solution into, always honoured when given.

        Returns
        -------
        DataContainer
            Current estimate of the solution.
        """
        if self.standard_form:
            return self.operator.map_to_solution(self.x, out=out)

        if out is not None:
            out.fill(self.x)
            return out
        return self.x
