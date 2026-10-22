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
# CIL Developers and contributers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt


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
import math

log = logging.getLogger(__name__)


class LSQR(Algorithm):

    r"""
    Least Squares with QR factorisation (LSQR) algorithm.

    The LSQR algorithm is used to solve large-scale linear systems and least-squares problems, particularly when the matrix is sparse or implicitly defined.

    Solves the problem:

    .. math::

        \min_x \|Ax - b\|_2^2

    Optionally, with generalised Tikhonov regularisation:

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_2^2

    where the structural operator :math:`L` can be provided as `struct_operator`
    and defaults to the identity. The regularised problem can be iterated on in
    two forms, chosen with `form`:

    * ``form='block'`` (the default) iterates on the stacked operator
      :math:`K = [A;\ \alpha L]` against :math:`[b;\ 0]`. It accepts any
      :math:`L`, penalises the solution :math:`\alpha^2\|Lx\|_2^2` from any
      starting point, and so warm starts, at the price of range containers
      spanning both :math:`\mathrm{Range}(A)` and :math:`\mathrm{Range}(L)`.

    * ``form='standard'`` changes variables to :math:`z = Lx` and iterates on
      :math:`K = A L^{-1}`, with :math:`\alpha` folded into LSQR's own scalar
      recurrence. It is cheaper -- no stacked containers, and for :math:`L = I`
      it costs the same as an unregularised solve -- but it requires an
      orthogonal, square :math:`L`, and LSQR cannot warm start in it: the
      penalty lands on the step rather than the solution, see the note below.

    Parameters
    ----------
    operator : Operator
        Linear operator representing the forward model.
    initial : DataContainer, optional
        Initial guess for the solution. If not provided, a zero-initialised container is used.
        In standard form a non-zero `initial` changes the objective, see the note below.
    data : DataContainer
        Measured data (right-hand side of the equation).
    alpha : float, optional
        Non-negative regularisation parameter. If zero, standard LSQR is used.
    struct_operator : Operator, optional
        The structural operator :math:`L`. ``None`` means the identity.
    form : {'block', 'standard'}, default 'block'
        Which formulation of the regularised problem to iterate on, see above.

    Note
    ----
    The note in this section applies to ``form='standard'`` only; the default
    block form penalises the solution from any starting point.

    In standard form, given a non-zero initial guess :math:`x_0`, LSQR solves for the update
    :math:`\delta = x - x_0` against the initial residual :math:`r_0 = b - Ax_0`. The scalar
    :math:`\alpha` is applied to whichever variable is being solved for, so it penalises
    :math:`\delta` and, for :math:`L = I`, the algorithm minimises

    .. math::

        \min_\delta \|A\delta - r_0\|_2^2 + \alpha^2 \|\delta\|_2^2
        \quad\Longleftrightarrow\quad
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|x - x_0\|_2^2 .

    This is Tikhonov regularisation towards :math:`x_0`, which is useful when a prior
    reconstruction is available, and it coincides with the :math:`\alpha^2 \|x\|_2^2` penalty
    only when :math:`x_0 = 0`. Since the two objectives differ, a warning is raised when a
    non-zero `initial` is combined with a non-zero `alpha` in standard form. To penalise the
    solution itself from a non-zero starting point, use the default ``form='block'``.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    """

    def __init__(self, initial=None, operator=None, data=None, alpha=0, struct_operator=None,
                 form='block', **kwargs):
        """
        Initialise the LSQR algorithm.

        Parameters
        ----------
        initial : DataContainer, optional
            Initial guess for the solution. When `alpha` is non-zero it also sets the point the
            penalty is applied relative to, see the note in the class documentation.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        alpha : float, optional
            Regularisation parameter. Default is 0 (no regularisation).
        struct_operator : Operator, optional
            Structured operator for regularisation.
        form : {'block', 'standard'}, default 'block'
            Which formulation of the regularised problem to iterate on. See
            the class documentation.
        """

        super(LSQR, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        self.regalpha = alpha

        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data,
                        struct_operator=struct_operator, form=form)
        else:
            raise ValueError(
                ' You must initialise LSQR with an `operator` and `data`')

    def set_up(self, initial, operator, data, struct_operator=None,
               form='block'):
        """
        Set up the LSQR algorithm with the problem definition.

        Allocates the entire workspace; nothing allocates afterwards.

        Parameters
        ----------
        initial : DataContainer
            Initial guess for the solution. When `alpha` is non-zero it also sets the point the
            penalty is applied relative to, see the note in the class documentation.
        operator : Operator
            Linear operator representing the forward model.
        data : DataContainer
            Measured data.
        struct_operator : Operator, optional
            Structured operator for regularisation.
        form : {'block', 'standard'}, default 'block'
            Which formulation to iterate on.
        """
        log.info("%s setting up", self.__class__.__name__)

        self.operator = _build_tikhonov_operator(
            operator, struct_operator=struct_operator,
            regalpha=self.regalpha, form=form)

        # The operator's class records which form was built; alpha == 0
        # leaves the bare operator and both flags False.
        self.standard_form = isinstance(self.operator, TikhonovOperator)
        self.block_form = isinstance(self.operator, BlockTikhonovOperator)

        if self.standard_form and initial is not None and initial.norm() > 0:
            warnings.warn(
                "LSQR in standard form with a non-zero `initial` minimises "
                "||A(x0 + d) - b||^2 + alpha^2 ||W L d||^2: the penalty applies "
                "to the step, not to the solution, so the result is not the "
                "regularised minimiser. Pass form='block' to warm start, or "
                "leave `initial` at zero.",
                UserWarning, stacklevel=2)

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
        self.v = self.x.copy()
        self.d = self.x.copy()
        self.tmp_iterate = self.x.copy()

        if self.block_form:
            self.u = BlockDataContainer(
                data.copy(), self.operator.struct_operator.direct(initial))
        else:
            self.u = data.copy()
        self.tmp_range = self.u.copy()

        self.initialise_variables()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @property
    def supports_warm_start(self):
        """False for standard-form LSQR, which damps the step rather than the
        solution; block form warm starts like any other."""
        return not self.standard_form

    def solution_geometry(self):
        """The geometry of the physical solution space."""
        if self.standard_form:
            return self.operator.solution_geometry()
        return self.operator.domain_geometry()

    def initialise_variables(self):
        """Initialise the variables of the algorithm. Allocates nothing."""
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

        # u = (data - operator * x) / beta
        self.operator.direct(self.x, out=self.u)
        if self.block_form:
            # Block 0: u[0] = data - operator[0] * x
            self.data.sapyb(1.0, self.u[0], -1.0, out=self.u[0])
            # Block 1: u[1] = 0.0 - operator[1] * x
            self.u[1].multiply(-1.0, out=self.u[1])
        else:
            # Standard single container form: u = data - A * x
            self.data.sapyb(1.0, self.u, -1.0, out=self.u)
        self.beta = self.u.norm()
        if self.beta > 0:
            # Reachable under IRLS: a converged warm start has zero residual.
            self.u /= self.beta

        # v = operator^* u / alpha
        self.operator.adjoint(self.u, out=self.v)
        self.alpha = self.v.norm()
        if self.alpha > 0:
            self.v /= self.alpha

        self.rhobar = self.alpha
        self.phibar = self.beta
        self.normr = self.beta
        self.res2 = 0.0
        self.regalphasq = self.regalpha**2

        self.d.fill(self.v)

    def update(self):
        """Perform a single iteration of the LSQR algorithm."""
        # Update u in GKB
        self.operator.direct(self.v, out=self.tmp_range)
        self.tmp_range.sapyb(1.,  self.u, -self.alpha, out=self.u)
        self.beta = self.u.norm()
        if self.beta == 0:
            # Golub-Kahan has terminated: the iterate is already the exact
            # solution, and dividing by zero would turn every later iterate into nan.
            raise StopIteration
        self.u /= self.beta

        # Update v in GKB
        self.operator.adjoint(self.u, out=self.tmp_iterate)
        self.v.sapyb(-self.beta, self.tmp_iterate, 1., out=self.v)
        self.alpha = self.v.norm()
        if self.alpha > 0:
            self.v /= self.alpha

        # Eliminate diagonal from regularisation
        if self.block_form or self.regalphasq == 0:
            # Nothing to eliminate. rhobar passes through untouched:
            # sqrt(rhobar**2) would flip the sign for rhobar < 0.
            rhobar1 = self.rhobar
            psi = 0
        else:
            # Standard form: alpha enters the scalar recurrence instead.
            rhobar1 = math.sqrt(self.rhobar * self.rhobar + self.regalphasq)
            c1 = self.rhobar / rhobar1
            s1 = self.regalpha / rhobar1
            psi = s1 * self.phibar
            self.phibar = c1 * self.phibar

        # Eliminate lower bidiagonal part
        rho = math.sqrt(rhobar1 ** 2 + self.beta ** 2)
        if rho == 0:
            # Both factors have collapsed; no step left to take.
            raise StopIteration
        c = rhobar1 / rho
        s = self.beta / rho
        theta = s * self.alpha
        self.rhobar = -c * self.alpha
        phi = c * self.phibar
        self.phibar = s * self.phibar

        # Update image x
        self.x.sapyb(1, self.d, phi/rho, out=self.x)

        # Update d
        self.d.sapyb(-theta/rho, self.v, 1, out=self.d)

        # Estimate residual norm
        self.res2 += psi ** 2
        self.normr = math.sqrt(self.phibar ** 2 + self.res2)

    def update_objective(self):
        """
        Update the objective function value (residual norm squared).
        """

        if self.normr is numpy.nan:
            raise StopIteration()
        self.loss.append(self.normr**2)

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
            Current estimate of the solution. With ``out=None``, standard form
            returns a freshly mapped container; block form returns the live
            iterate, not a copy.
        """
        if self.standard_form:
            return self.operator.map_to_solution(self.x, out=out)

        if out is not None:
            out.fill(self.x)
            return out
        return self.x
