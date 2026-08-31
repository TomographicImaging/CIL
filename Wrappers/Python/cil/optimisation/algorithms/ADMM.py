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

from cil.optimisation.algorithms import Algorithm
from cil.optimisation.utilities import StepSizeRule, LADMMConstantStepSize
from cil.optimisation.utilities.StepSizeMethods import _validate_ladmm_step_sizes
import logging
import warnings

log = logging.getLogger(__name__)


class LADMM(Algorithm):
    r"""
    LADMM is the Linearized Alternating Direction Method of Multipliers (LADMM).

    The general form of ADMM is given by the following optimization problem:

    .. math::

        \min_{x} f(x) + g(y), \text{ subject to } Ax + By = b

    In CIL, we have implemented the case where :math:`A = Id`, :math:`B = -K`, :math:`b = 0`  which gives

    .. math::

        \min_x f(Kx) + g(x).

    The algorithm is given by the following iteration, for :math:`k\geq 1`:

    .. math::

        \begin{cases}
            x_{k} = \mathrm{prox}_{\tau f} \left(x_{k-1} - \dfrac{\tau}{\sigma} A_{T}\left(Ax_{k-1} - z_{k-1} + u_{k-1} \right)  \right)\\
            z_{k} = \mathrm{prox}_{\sigma g} \left(Ax_{k} + u_{k-1}\right) \\
            u_{k} = u_{k-1} + Ax_{k} - z_{k}
        \end{cases}

    where :math:`\mathrm{prox}_{\tau f}` is the proximal operator of :math:`f` and :math:`\mathrm{prox}_{\sigma g}` is the proximal operator of :math:`g`.


    Parameters
    ------------
    operator:  CIL Linear Operator
        Operator :math:`K` in the objective function
    f: CIL Function
        Convex function with "simple" proximal
    g: CIL Function
        Convex function with "simple" proximal
    step_size: :class:`~cil.optimisation.utilities.StepSizeRule`, or `list` or `tuple` of length two, optional, default=None
        Either a LADMM compatible step size rule, or a pair :math:`(\tau, \sigma)` of the primal and dual step
        sizes. Either entry may be ``None``, in which case :math:`\sigma = 1` and :math:`\tau = \sigma/\|K\|^{2}`.
    initial: DataContainer, defaults to DataContainer filled with zeros
        Initial guess

    Other Parameters
    ----------------
    sigma: float, positive
        Deprecated. Pass as part of `step_size` instead.
    tau: float, positive
        Deprecated. Pass as part of `step_size` instead.



    Note
    -----
    Reference (Section 8) : O’Connor, D., Vandenberghe, L. On the equivalence of the primal-dual hybrid gradient method and Douglas–Rachford splitting. Math. Program. 179, 85–108 (2020). https://doi.org/10.1007/s10107-018-1321-1

    """

    def __init__(self, f=None, g=None, operator=None, step_size=None,
                       initial = None, **kwargs):

        """Initialisation of the algorithm."""

        sigma = kwargs.pop('sigma', None)  # To be deprecated
        tau = kwargs.pop('tau', None)  # To be deprecated

        if step_size is not None and (sigma is not None or tau is not None):  # To be deprecated
            raise ValueError("The parameters `sigma` and `tau` are being deprecated in favour of `step_size`. You have passed both. Instead please pass these as part of the `step_size` argument, either as a tuple of (tau, sigma) or using a compatible step size rule.")

        if sigma is not None or tau is not None:  # To be deprecated
            warnings.warn("The parameters `sigma` and `tau` are being deprecated. In the future, please pass these as part of the `step_size` argument, either as a tuple of (tau, sigma) or using a compatible step size rule.", category=DeprecationWarning, stacklevel=2)
            step_size = (tau, sigma)

        super(LADMM, self).__init__(**kwargs)

        self.set_up(f = f, g = g, operator = operator, step_size = step_size,
             initial=initial)

    @property
    def tau(self):
        """The primal step-size currently in use. May be updated at each iteration by a step size rule."""
        return self._tau

    @property
    def sigma(self):
        """The dual step-size currently in use. May be updated at each iteration by a step size rule."""
        return self._sigma

    def set_up(self, f, g, operator, step_size=[None, None], initial=None):
        """Set up of the algorithm."""
        log.info("%s setting up", self.__class__.__name__)

        self.f = f
        self.g = g
        self.operator = operator

        if step_size is None:  # This line can be removed when sigma and tau deprecated
            step_size = (None, None)
        if isinstance(step_size, StepSizeRule):
            if not hasattr(step_size, 'get_initial_step_size'):
                raise ValueError(
                    "The step-size rule {} does not provide initial primal/dual step sizes "
                    "and is not compatible with LADMM.".format(type(step_size).__name__))
            self.step_size_rule = step_size
        elif isinstance(step_size, (tuple, list)):
            self.step_size_rule = LADMMConstantStepSize(step_size=step_size)
        else:
            raise ValueError("The `step_size` argument must be either None, a LADMM compatible step size rule or a tuple of (tau, sigma) where sigma is the step size for the dual problem and tau is the step size for the primal problem.")

        if initial is None:
            self.x = self.operator.domain_geometry().allocate(0)
        else:
            self.x = initial.copy()

        # allocate space for operator direct & adjoint
        self.tmp_dir = self.operator.range_geometry().allocate(0)
        self.tmp_adj = self.operator.domain_geometry().allocate(0)

        self.z = self.operator.range_geometry().allocate(0)
        self.u = self.operator.range_geometry().allocate(0)

        # the first update uses tmp_dir as Kx, so it must hold Kx of the initial point
        self.operator.direct(self.x, out=self.tmp_dir)

        self._tau, self._sigma = self.step_size_rule.get_initial_step_size(self)
        _validate_ladmm_step_sizes(self._tau, self._sigma, self.operator)

        self.configured = True

        log.info("%s configured", self.__class__.__name__)

    def update(self):
        """Performs a single iteration of the LADMM algorithm.
        """
        self.tmp_dir += self.u
        self.tmp_dir -= self.z
        self.operator.adjoint(self.tmp_dir, out = self.tmp_adj)

        self.x.sapyb(1, self.tmp_adj, -(self.tau/self.sigma), out=self.x)

        # apply proximal of f
        tmp = self.f.proximal(self.x, self.tau)
        self.operator.direct(tmp, out=self.tmp_dir)
        # store the result in x
        self.x.fill(tmp)
        del tmp

        self.u += self.tmp_dir

        # apply proximal of g
        self.g.proximal(self.u, self.sigma, out = self.z)

        # update
        self.u -= self.z

        tau, sigma = self.step_size_rule.get_step_size(self)
        if sigma is not self._sigma:
            self.u *= sigma / self._sigma #self.u = self.u/rho where rho is the adaptive penalty factor
        self._tau, self._sigma = tau, sigma

    def update_objective(self):
        """Update the objective function value"""
        self.loss.append(self.f(self.x) +  self.g(self.operator.direct(self.x)) )
