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


from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators.TikhonovOperator import (
    create_tikhonov_operator,
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

    Optionally, with Tikhonov regularisation towards the initial guess :math:`x_0`:

    .. math::

        \min_x \|Ax - b\|_2^2 + \alpha^2 \|x - x_0\|_2^2

    which reduces to the usual :math:`\alpha^2 \|x\|_2^2` penalty for the default zero initial.
    See the note below.

    1. Structured Tikhonov regularisation

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_2^2

    For structured regularisation, the operator :math:`L` can be provided as `struct_operator`. If not provided, it defaults to the identity operator.

    2. Sparsity-promoting regularisation (via IRLS algorithm)

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_1

    equivalent to solving iteratively

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|W_k Lx\|_2^2

    Where :math:`W_k` is a diagonal weight matrix that is updated at each outer iteration.

    Parameters
    ----------
    operator : Operator
        Linear operator representing the forward model.
    initial : DataContainer, optional
        Initial guess for the solution. If not provided, a zero-initialised container is used.
        When `alpha` is non-zero it also sets the point the penalty is applied relative to, see
        the note below.
    data : DataContainer
        Measured data (right-hand side of the equation).
    alpha : float, optional
        Non-negative regularisation parameter. If zero, standard LSQR is used. Otherwise the
        penalty is applied relative to `initial`, see the note below.

    Note
    ----
    Passing a non-zero `alpha` gives the option for Tikhonov regularisation without building a
    block operator and data container, and consequently at a lower memory cost.

    Given a non-zero initial guess :math:`x_0`, LSQR solves for the update
    :math:`\delta = x - x_0` against the initial residual :math:`r_0 = b - Ax_0`. The scalar
    :math:`\alpha` is applied to whichever variable is being solved for, so it penalises
    :math:`\delta` and the algorithm minimises

    .. math::

        \min_\delta \|A\delta - r_0\|_2^2 + \alpha^2 \|\delta\|_2^2
        \quad\Longleftrightarrow\quad
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|x - x_0\|_2^2 .

    This is Tikhonov regularisation towards :math:`x_0`, which is useful when a prior
    reconstruction is available, and it coincides with the :math:`\alpha^2 \|x\|_2^2` penalty
    only when :math:`x_0 = 0`. Since the two objectives differ, a warning is raised when a
    non-zero `initial` is combined with a non-zero `alpha`.

    To penalise :math:`\|x\|_2^2` from a non-zero starting point, build the block system
    explicitly and pass it to an unregularised LSQR (or to
    :class:`~cil.optimisation.algorithms.CGLS`):

    .. code-block:: python

        from cil.framework import BlockDataContainer
        from cil.optimisation.operators import BlockOperator, IdentityOperator

        block_operator = BlockOperator(A, alpha * IdentityOperator(A.domain_geometry()))
        block_data = BlockDataContainer(b, A.domain_geometry().allocate(0))
        lsqr = LSQR(initial=x0, operator=block_operator, data=block_data)

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/lsqr/
    """

    #: False: in standard form LSQR eliminates :math:`\alpha` with a Givens
    #: rotation on the bidiagonalisation of the residual of :math:`x_0`, so the
    #: penalty falls on the step and not on the solution. A class attribute
    #: rather than a property, because
    #: :func:`~cil.optimisation.operators.TikhonovOperator.resolve_form` needs
    #: it *before* an instance exists to have a form at all.
    warm_starts_in_standard_form = False

    def __init__(self, initial=None, operator=None, data=None, alpha=0, struct_operator=None,
                 form='auto', weighted=False, **kwargs):
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
        form : {'auto', 'standard', 'block'}, default 'auto'
            Which formulation of the regularised problem to iterate on. See
            :func:`~cil.optimisation.operators.TikhonovOperator.create_tikhonov_operator`.
        weighted : bool, default False
            Allocate the IRLS weight operator up front. Set this when the solver
            will be driven by :class:`~cil.optimisation.algorithms.IRLS`, so that
            the weights are part of the ``set_up`` budget rather than appearing
            mid-solve. It also tells ``form='auto'`` that a reweighting loop is
            coming, so it takes the block form first time rather than leaving
            IRLS to correct it through :meth:`rebuild_in_block_form`.
        """

        super(LSQR, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        self.regalpha = alpha

        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data,
                        struct_operator=struct_operator, form=form,
                        weighted=weighted)
        else:
            raise ValueError(
                ' You must initialise LSQR with an `operator` and `data`')

    def set_up(self, initial, operator, data, struct_operator=None,
               form='auto', weighted=False):
        """
        Set up the LSQR algorithm with the problem definition.

        Allocates the entire workspace. Neither :meth:`initialise_variables` nor
        :meth:`update` allocates anything afterwards, so an IRLS outer loop can
        re-enter them indefinitely at constant memory.

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
        form : {'auto', 'standard', 'block'}, default 'auto'
            Which formulation to iterate on.
        weighted : bool, default False
            Allocate the IRLS weight operator up front.
        """
        log.info("%s setting up", self.__class__.__name__)

        # What was asked for, as opposed to what it resolved to, and the
        # arguments needed to ask again. Only references to objects already held,
        # so this costs nothing. See rebuild_in_block_form.
        self.requested_form = form
        self._setup_args = dict(operator=operator, data=data,
                                struct_operator=struct_operator)

        # `solver=self` is read for the class attribute
        # warm_starts_in_standard_form, not for the `supports_warm_start`
        # property: that one is derived from the form, which is what this call
        # is deciding.
        self.operator = create_tikhonov_operator(
            operator, operator.domain_geometry(), struct_operator,
            self.regalpha, form=form, weighted=weighted, solver=self,
            zero_initial=initial is None or initial.norm() == 0)

        # The resolved form, as a string, so nothing downstream has to isinstance
        # its way through the class hierarchy. 'none' when alpha == 0 and the
        # bare operator is used unwrapped.
        self.form = getattr(self.operator, 'form', 'none')
        self.standard_form = self.form == 'standard'
        self.block_form = self.form == 'block'

        if self.standard_form and initial is not None and initial.norm() > 0:
            warnings.warn(
                "LSQR in standard form with a non-zero `initial` minimises "
                "||A(x0 + d) - b||^2 + alpha^2 ||W L d||^2: the penalty applies "
                "to the step, not to the solution, so the result is not the "
                "regularised minimiser. Pass form='block' to warm start, or "
                "leave `initial` at zero.",
                UserWarning, stacklevel=2)

        # Set pointer to the data container
        self.data = data

        # Allocate 4 domain containers for the LSQR algorithm. In block form the
        # domain is the image space; in standard form it is Range(L), which is
        # not the same space and need not be the same size.
        self.initial = initial
        self.x = self.operator.domain_geometry().allocate(0)
        self.v = self.operator.domain_geometry().allocate(0)
        self.d = self.operator.domain_geometry().allocate(0)
        self.tmp_iterate = self.operator.domain_geometry().allocate(0)

        # Allocate 2 range containers for the LSQR algorithm
        self.u = self.operator.range_geometry().allocate(0)
        self.tmp_range = self.operator.range_geometry().allocate(0)

        # Initialize the state of the algorithm
        self.initialise_variables()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @property
    def supports_warm_start(self):
        """
        False for standard-form LSQR, which damps the step rather than the
        solution and so cannot resume from a non-zero iterate.

        The block form carries the penalty as a row of :math:`K`, where it
        applies to the solution, and warm starts like any other.
        """
        return self.warm_starts_in_standard_form or not self.standard_form

    def rebuild_in_block_form(self):
        r"""
        Re-run :meth:`set_up` with ``form='block'``, if ``'auto'`` chose
        otherwise.

        ``form='auto'`` is resolved in ``set_up``, which runs from the
        constructor -- before an :class:`~cil.optimisation.algorithms.IRLS`
        outer loop can exist to be asked about. From a zero start it therefore
        takes the cheaper standard form, which is the right answer for one solve
        and the wrong one from the second solve of a reweighting loop onwards,
        where LSQR resumes from the previous iterate and damps the step rather
        than the solution. IRLS calls this when it attaches, so the choice ends
        up being made on what actually happens rather than on what was knowable
        at construction.

        Only ``form='auto'`` is overridden. An explicit ``form='standard'`` is
        the caller's decision and is left alone.

        This reallocates the whole workspace and resets the weights to one, so
        it is a construction-time correction and not something to call
        mid-solve. Passing ``weighted=True`` to the constructor avoids it
        altogether, ``'auto'`` then having what it needs first time.

        Returns
        -------
        bool
            Whether the form changed.
        """
        if self.requested_form != 'auto' or self.block_form:
            return False

        log.info("Rebuilding %s in block form: it cannot warm start in the "
                 "standard form, and an outer loop has attached to it.",
                 self.__class__.__name__)
        self.set_up(initial=self.initial, form='block',
                    weighted=self.weights is not None, **self._setup_args)
        # Not unconditionally True: with regalpha == 0 there is no penalty to
        # place anywhere and the bare operator is used whatever `form` says.
        return self.block_form

    @property
    def weights(self):
        """The mutable IRLS weights, or ``None`` when unweighted."""
        return getattr(self.operator, 'weights', None)

    def enable_weights(self):
        """
        Allocate the IRLS weights after the fact.

        Prefer ``weighted=True`` at construction, which keeps every allocation
        inside ``set_up``. This exists for an IRLS instance attached to an
        already-built solver.
        """
        return self.operator.enable_weights()

    def solution_geometry(self):
        """
        The geometry of the physical solution space, whatever the form.

        In block form (and unregularised) this is the solver's own domain; in
        standard form the iterate lives in ``Range(L)`` and the solution is
        recovered through :math:`(WL)^{-1}`, whose range this is.
        """
        if self.standard_form:
            return self.operator.reg_operator.domain_geometry()
        return self.operator.domain_geometry()

    def initialise_variables(self):
        """
        Initialise the variables of the algorithm.

        Allocates nothing: every container written here was allocated in
        :meth:`set_up`.
        """
        # alpha lives in two places: this solver's scalar recurrence, and the
        # operator handed back by the factory. set_up sets them together from
        # the same value; catch anyone who has moved one since, before a whole
        # solve quietly runs on the pair disagreeing.
        operator_alpha = getattr(self.operator, 'regalpha', None)
        if operator_alpha is not None and operator_alpha != self.regalpha:
            raise ValueError(
                "self.regalpha ({}) and self.operator.regalpha ({}) disagree. "
                "Change both together, or re-run set_up.".format(
                    self.regalpha, operator_alpha))

        # Map initial guess to the solution space
        if self.standard_form:
            # Standard form operates in structure space: x_0 = W * L * u_0
            self.operator.reg_operator.direct(self.initial, out=self.x)
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

        # Initialise scalars
        self.rhobar = self.alpha
        self.phibar = self.beta
        self.normr = self.beta
        self.res2 = 0.0
        self.regalphasq = self.regalpha**2

        # Initialise the search direction
        self.d.fill(self.v)

    def update(self):
        """Perform a single iteration of the LSQR algorithm."""
        # Update u in GKB
        self.operator.direct(self.v, out=self.tmp_range)
        self.tmp_range.sapyb(1.,  self.u, -self.alpha, out=self.u)
        self.beta = self.u.norm()
        if self.beta == 0:
            # Golub-Kahan has terminated: the Krylov subspace is exhausted and
            # the current iterate is already the exact solution. Dividing by
            # zero would put inf into u and turn every later iterate into nan,
            # silently -- and the case is easy to hit: K = [A; alpha*W] with A
            # and W both multiples of the identity has K^T K a multiple of the
            # identity, so LSQR converges in one step and the second breaks
            # down. The guard on alpha below is the same case on the other
            # factor.
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
            # Nothing to eliminate: in block form the penalty is already a row
            # of K, and unregularised there is no penalty at all. rhobar goes
            # through untouched -- sqrt(rhobar**2) is not the identity: it
            # drops a rounding bit, and for rhobar < 0 (reachable, since
            # rhobar = -c*alpha) it would flip the sign of phibar.
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
            # Both the bidiagonal and the eliminated diagonal have collapsed;
            # there is no step left to take.
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

        if numpy.isnan(self.normr):
            raise StopIteration()
        self.loss.append(self.normr**2)

    def get_output(self, out=None):
        """
        Get the current solution estimate, in the physical solution space.

        Parameters
        ----------
        out : DataContainer, optional
            Buffer to write the solution into, always honoured when given.
            Pass one from a loop that calls this repeatedly, such as IRLS: it
            makes the result an independent snapshot in both forms, and in
            standard form -- where the iterate lives in ``Range(L)`` and has
            to be mapped back through :math:`(WL)^{-1}` -- it also avoids an
            allocation per call.

        Returns
        -------
        DataContainer
            Current estimate of the solution. With ``out=None`` the two forms
            differ: standard form returns a freshly mapped container, while
            block form returns the **live iterate**, not a copy -- the
            convention :meth:`Algorithm.get_output` sets. Anything held across
            iterations should come through ``out=`` or be copied.
        """
        if self.standard_form:
            return self.operator.reg_operator.inverse(self.x, out=out)

        if out is not None:
            out.fill(self.x)
            return out
        return self.x
