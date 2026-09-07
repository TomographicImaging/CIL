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

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.operators.TikhonovOperator import (
    create_tikhonov_operator,
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

    3. Sparsity-promoting regularisation (via IRLS algorithm)

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_1

    equivalent to solving iteratively

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|W_k Lx\|_2^2

    Where :math:`W_k` is a diagonal weight matrix that is updated at each outer iteration.

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
    form : {'auto', 'standard', 'block'}, default 'auto'
        Which formulation of the regularised problem to iterate on. See
        :func:`~cil.optimisation.operators.TikhonovOperator.create_tikhonov_operator`.
    weighted : bool, default False
        Allocate the IRLS weight operator up front, keeping it inside the
        ``set_up`` budget.

    Reference
    ---------
    https://web.stanford.edu/group/SOL/software/cgls/
    '''

    #: True: CGLS subtracts the :math:`\alpha^2 x` term explicitly, using the
    #: whole iterate, so it runs on the true regularised normal equations in
    #: either form. A class attribute rather than a property, because
    #: :func:`~cil.optimisation.operators.TikhonovOperator.resolve_form` needs
    #: it *before* an instance exists to have a form at all.
    warm_starts_in_standard_form = True

    def __init__(self, initial=None, operator=None, data=None, alpha=0, struct_operator=None,
                 form='auto', weighted=False, **kwargs):
        '''initialisation of the algorithm
        '''
        super(CGLS, self).__init__(**kwargs)

        if initial is None and operator is not None:
            initial = operator.domain_geometry().allocate(0)
        self.regalpha = alpha

        if initial is not None and operator is not None and data is not None:
            self.set_up(initial=initial, operator=operator, data=data,
                        struct_operator=struct_operator, form=form,
                        weighted=weighted)

    def set_up(self, initial, operator, data, struct_operator=None,
               form='auto', weighted=False):
        r'''Initialisation of the algorithm

        Allocates the entire workspace. Neither :meth:`initialise_variables` nor
        :meth:`update` allocates anything afterwards, so an IRLS outer loop can
        re-enter them indefinitely at constant memory.

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
        form : {'auto', 'standard', 'block'}, default 'auto'
            Which formulation of the regularised problem to iterate on. See
            :func:`~cil.optimisation.operators.TikhonovOperator.create_tikhonov_operator`.
        weighted : bool, default False
            Allocate the IRLS weight operator up front, keeping it inside the
            ``set_up`` budget.

        '''

        log.info("%s setting up", self.__class__.__name__)

        # What was asked for, as opposed to what it resolved to. CGLS never
        # needs to revisit the choice -- see supports_warm_start -- but IRLS
        # reads this off whichever inner solver it is given.
        self.requested_form = form

        # `solver=self` carries warm_starts_in_standard_form = True, which lets
        # form='auto' take the cheaper standard form whenever (WL)^-1 exists,
        # IRLS included: CGLS subtracts the alpha^2 x term itself, so the
        # starting point cannot move the minimiser.
        self.operator = create_tikhonov_operator(
            operator, operator.domain_geometry(), struct_operator,
            self.regalpha, form=form, weighted=weighted, solver=self)

        self.form = getattr(self.operator, 'form', 'none')
        self.standard_form = self.form == 'standard'
        self.block_form = self.form == 'block'

        # Set pointer to the data container
        self.data = data

        # Allocate Domain variables
        self.initial = initial
        self.x = self.operator.domain_geometry().allocate(0)
        self.s = self.operator.domain_geometry().allocate(0)
        self.p = self.operator.domain_geometry().allocate(0)

        # Allocate Range variables
        self.r = self.operator.range_geometry().allocate(0)
        self.q = self.operator.range_geometry().allocate(0)

        # Initialize the variables
        self.initialise_variables()

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    @property
    def supports_warm_start(self):
        """
        Always True.

        Unlike LSQR, CGLS subtracts the :math:`\\alpha^2 x` term explicitly in
        :meth:`initialise_variables` and :meth:`update`, so it runs on the true
        regularised normal equations in either form and the starting point
        cannot move the minimiser.
        """
        return self.warm_starts_in_standard_form or not self.standard_form

    @property
    def weights(self):
        """The mutable IRLS weights, or ``None`` when unweighted."""
        return getattr(self.operator, 'weights', None)

    def enable_weights(self):
        """
        Allocate the IRLS weights after the fact.

        Prefer ``weighted=True`` at construction, which keeps every allocation
        inside ``set_up``.
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
        r'''
        Initialise the variables of the algorithm.

        Allocates nothing: every container written here was allocated in
        :meth:`set_up`.
        '''
        # alpha lives in two places: this solver's normal-equations term, and
        # the operator handed back by the factory. set_up sets them together
        # from the same value; catch anyone who has moved one since, before a
        # whole solve quietly runs on the pair disagreeing.
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

        # Initialize the search direction
        self.p.fill(self.s)

        # Initialize the norms
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
        if numpy.isnan(a):
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
