# Copyright 2026 United Kingdom Research and Innovation Copyright 2026 The
# University of Manchester Licensed under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0 Unless required by applicable law
# or agreed to in writing, software distributed under the License is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License. Authors: - CIL
# Developers, listed at:
# https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

"""The adaptive PDHG step-size rule of :cite:`Goldstein2015`.

This rule is kept here for reference and is not part of the public CIL API: it
is not exported from :mod:`cil.optimisation.utilities` and is not documented.
In our experiments
:class:`~cil.optimisation.utilities.PDHGAdaptiveStepSize2013` performed more
consistently. It is intended to move to the CIL-Demos ``misc`` folder; it is
kept in CIL, with its tests, until then so that it stays up to date with the
PDHG step-size interface.

Import it with::

    from cil.optimisation.utilities.pdhg_adaptive_step_size_2015 import PDHGAdaptiveStepSize2015
"""

import logging

import numpy as np

from cil.optimisation.utilities.StepSizeMethods import PDHGStepSizeRule

log = logging.getLogger(__name__)


class PDHGAdaptiveStepSize2015(PDHGStepSizeRule):
    r"""Adaptively updates the PDHG primal and dual step sizes using the
    backtracking and residual-balancing method of :cite:`Goldstein2015`.

    This is a PDHG-compatible step-size rule and a variant of
    :class:`PDHGAdaptiveStepSize2013`. It provides an initial pair of step
    sizes via :meth:`get_initial_step_size` and then, at the end of every PDHG
    iteration, :meth:`get_step_size` updates :math:`\tau` and :math:`\sigma` in
    place. Two mechanisms are applied each iteration.

    **1. Backtracking.** With the increments :math:`\Delta x = x^{n+1} - x^{n}`
    and :math:`\Delta y = y^{n+1} - y^{n}`, a backtracking quantity is formed

    .. math::

        b = c\,\sigma\,\|\Delta x\|^{2} + c\,\tau\,\|\Delta y\|^{2}
            - 4\,\sigma\tau\,|\langle \Delta y, K\Delta x\rangle|.

    The step is accepted when :math:`b \ge 0`. Otherwise both step sizes are
    halved, :math:`\tau \leftarrow \tfrac{1}{2}\tau`, :math:`\sigma \leftarrow
    \tfrac{1}{2}\sigma`, the PDHG update is recomputed (via
    :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update`) and :math:`b` is
    re-evaluated, up to ``inner_iterations`` times. Because
    :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update` overwrites ``x`` and
    ``y`` in place, the rule keeps a copy of the primal and dual iterates as
    they were at the start of the iteration and restores both before each
    retry, so that every trial step is retaken from the same starting point
    rather than from the rejected one.

    **2. Residual balancing.** The primal and dual residuals

    .. math::

        p = \left\| \tfrac{\Delta x}{\tau} - K^{*}\Delta y \right\|, \qquad
        d = \|K\| \left\| \tfrac{\Delta y}{\sigma} - K\Delta x \right\|

    are compared. If :math:`2p < d` the ratio is tilted towards the dual
    variable (:math:`\tau \leftarrow (1-\alpha)\tau`, :math:`\sigma \leftarrow
    \sigma/(1-\alpha)`); if :math:`2d < p` it is tilted towards the primal
    variable (:math:`\tau \leftarrow \tau/(1-\alpha)`, :math:`\sigma \leftarrow
    (1-\alpha)\sigma`). Whenever a tilt is applied the adaptation strength
    decays, :math:`\alpha \leftarrow \eta\,\alpha`. The balancing step
    preserves the product :math:`\sigma\tau`.

    Parameters
    -------------
    initial_step_size : list of two positive :obj:`float`, optional, default= [10/algorithm.operator.norm(), 10/algorithm.operator.norm()]
        Initial values of the primal and dual step sizes used in the adaptive
        step size method.
    initial_alpha : :obj:`float` in :math:`(0,1)`, optional, default=0.95
        Initial value of the adaptation strength :math:`\alpha` controlling the
        size of the residual-balancing update. Each rebalance multiplies or
        divides :math:`\tau` by :math:`1-\alpha`, so :math:`\alpha` close to
        one gives a *large* rescaling (:math:`\alpha = 0.95` changes
        :math:`\tau` by a factor of 20) and :math:`\alpha` close to zero a
        gentle one.
    eta : positive :obj:`float`, optional, default=0.95
        The decay factor :math:`\eta \in (0,1)` applied to :math:`\alpha` each
        time the step sizes are rebalanced.
    c : positive :obj:`float`, optional, default=0.9
        The convergence constant :math:`c` appearing in the backtracking
        quantity :math:`b`.
    inner_iterations : :obj:`int`, optional, default=50
        The maximum number of inner iterations for the backtracking loop.
    auto_stop : :obj:`boolean`, optional, default=True
        If True, the adaptive step size method automatically stops updating the
        step sizes once they have not changed for more than
        ``auto_stop_patience`` consecutive iterations.
    auto_stop_patience : :obj:`int`, optional, default=10
        Number of consecutive iterations with no change to the step sizes that
        are tolerated before the adaptive updates are stopped (only used when
        ``auto_stop=True``).


    Notes
    -----
    This method is memory expensive, requiring the storage of 2 extra image
    copies and 2 extra data copies: the primal and dual residuals, and the
    primal and dual iterates as they were at the start of the iteration, which
    are needed to retake a rejected backtracking step. When ``auto_stop=True``
    the adaptive updates are switched off once the step sizes have been
    unchanged for more than ``auto_stop_patience`` consecutive iterations; all
    four are then released, reducing ongoing memory use and computational cost.
    For a more time expensive, but less memory expensive method, see
    :class:`PDHGBayesOptimisationStepSize` which does not require the storage
    of extra images and data.

    Both mechanisms are skipped for an iteration in which the primal and dual
    residuals have both fallen below the attribute ``tolerance`` (default
    ``1e-6``), on the grounds that the iterates have stopped moving and the
    residuals no longer carry usable information. ``tolerance`` is not a
    constructor argument but can be set on the rule after construction.

    See Also
    --------
    PDHGAdaptiveStepSize2013 : The related backtracking/residual-balancing rule
    from :cite:`goldstein2013adaptive`.

    Reference
    ---------
    Goldstein, T., Li, M. and Yuan, X., 2015. Adaptive primal-dual splitting
    methods for statistical learning and image processing. Advances in Neural
    Information Processing Systems, 28. :cite:`Goldstein2015`
    """

    enforces_convergence = True

    def __init__(self, initial_step_size=(None, None), initial_alpha=0.95,
                 eta=0.95, c=0.9, inner_iterations=50, auto_stop=True,
                 auto_stop_patience=10):
        '''Initialises the step size rule'''

        self.adaptive = True
        self.alpha = initial_alpha
        self.eta = eta
        self.c = c
        self.tolerance = 1e-6
        self.p_norm = 100
        self.d_norm = 100
        self.auto_stop = auto_stop
        self.auto_stop_patience = auto_stop_patience
        self.count = 0
        self.inner_iterations = inner_iterations

        self.y_old = None
        self.x_resid = None
        self.y_resid = None
        self.x_prev = None
        self.initial_step_size = initial_step_size
        if len(initial_step_size) != 2:
            raise ValueError(
                "initial_step_size should be a list or tuple of length two, "
                "step_size = {}".format(initial_step_size))

    def get_initial_step_size(self, algorithm):
        r"""Returns the initial primal and dual step sizes for the PDHG
        algorithm.

        Values are taken from ``initial_step_size``; each entry left as
        ``None`` defaults to :math:`10/\|K\|`.
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to, used here for its operator
            and its initial dual variable.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        tau = self.initial_step_size[0]
        sigma = self.initial_step_size[1]
        if tau is None:
            tau = 10/algorithm.operator.norm()
        if sigma is None:
            sigma = 10/algorithm.operator.norm()
        # y^{0}, the dual iterate a backtracking retry in the first iteration
        # must be restored to. It is captured here, during PDHG.set_up, because
        # by the time get_step_size is first called algorithm.y already holds
        # y^{1}.
        self.y_old = algorithm.y.copy()  # Extra range data 1
        return tau, sigma

    def get_step_size(self, algorithm):
        """Applies the backtracking and residual-balancing updates for one PDHG
        iteration.

        Both steps modify ``algorithm._tau`` and ``algorithm._sigma`` in place,
        and backtracking additionally recomputes the PDHG update. Once the
        adaptive updates have been stopped by ``auto_stop`` this returns the
        current step sizes unchanged.

        Parameters
        ----------
        algorithm : :class:`~cil.optimisation.algorithms.PDHG`
            The algorithm the rule is attached to.

        Returns
        -------
        tuple
            the primal and dual step sizes as a tuple ``(tau, sigma)``
        """
        if self.adaptive:
            if self.x_prev is None:
                if self.y_old is None:
                    self.y_old = algorithm.operator.range_geometry().allocate(
                        0
                    )  # Extra range data 1
                self.x_resid = algorithm.operator.domain_geometry().allocate(
                    0
                )  # Extra image 1
                self.y_resid = algorithm.operator.range_geometry().allocate(
                    0
                )  # Extra range data 2
                # x^{k-1}, the extrapolation partner of this iteration. On the
                # first call x^{-1} = x^{0} = algorithm.x_old, which is what
                # set_up copied x from.
                self.x_prev = algorithm.x_old.copy()  # Extra image 2
            if self.p_norm > self.tolerance and self.d_norm > self.tolerance:
                log.debug(
                    'Before adaptive step-size step, tau = {}, sigma = '
                    '{}'.format(
                        algorithm._tau, algorithm._sigma
                    )
                )

                b = self._calculate_backtracking(algorithm)
                converged = False
                for k in range(self.inner_iterations):
                    if b >= 0:
                        log.debug(
                            'Finished backtracking step, backtracking '
                            'value b = {}, step sizes are tau = {}, '
                            'sigma = {}'.format(
                                b, algorithm._tau, algorithm._sigma))
                        converged = True
                        break
                    algorithm._tau *= 0.5
                    algorithm._sigma *= 0.5
                    log.debug(
                        ' Backtracking step - multiplying primal and dual '
                        'step sizes by 1/2, new step sizes are tau = {}, '
                        'sigma ={}'.format(
                            algorithm._tau, algorithm._sigma))

                    self._restore_iterate(algorithm)
                    algorithm._pdhg_update()
                    b = self._calculate_backtracking(algorithm)
                    self.count = 0
                if not converged:
                    log.warning(
                        'Backtracking step did not converge after {} '
                        'iterations, backtracking value b = {}, step sizes '
                        'are tau = {}, sigma = {}'.format(
                            self.inner_iterations,
                            b,
                            algorithm._tau,
                            algorithm._sigma,
                        )
                    )

                self._calculate_pnorm_dnorm(algorithm)
                log.debug(
                    'Started the rebalancing step with p_norm = {}, d_norm = '
                    '{}'.format(
                        self.p_norm, self.d_norm))
                if 2*self.p_norm < self.d_norm:
                    algorithm._tau *= (1 - self.alpha)
                    algorithm._sigma /= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                    log.debug(
                        '2*p_norm < d_norm so rebalancing step sizes, new '
                        'step sizes are tau = {}, sigma ={}'.format(
                            algorithm._tau, algorithm._sigma))
                elif 2*self.d_norm < self.p_norm:
                    algorithm._tau /= (1 - self.alpha)
                    algorithm._sigma *= (1 - self.alpha)
                    self.alpha *= self.eta
                    self.count = 0
                    log.debug(
                        '2*d_norm < p_norm so rebalancing step sizes, new '
                        'step sizes are tau = {}, sigma ={}'.format(
                            algorithm._tau, algorithm._sigma))
                else:
                    log.debug(
                        'No change from the rebalancing step, step sizes are '
                        'tau = {}, sigma ={}'.format(
                            algorithm._tau, algorithm._sigma))
                    self.count += 1
            else:
                log.debug(
                    'No change from the rebalancing step as pnorm and dnorm '
                    'are below threshold, step sizes are tau = {}, '
                    'sigma ={}'.format(algorithm._tau, algorithm._sigma))
            self.y_old.fill(algorithm.y)
            # algorithm.x_old is x^{k}, the extrapolation partner of iteration
            # k+1
            self.x_prev.fill(algorithm.x_old)

            if self.count > self.auto_stop_patience and self.auto_stop:
                self.adaptive = False
                log.debug(
                    'Automatic stopping of adaptive step size updates, step '
                    'sizes have not changed for {} iterations, step sizes '
                    'are tau = {}, sigma ={}'.format(
                        self.auto_stop_patience, algorithm._tau,
                        algorithm._sigma))
                del self.y_resid
                del self.y_old
                del self.x_resid
                del self.x_prev

        return algorithm._tau, algorithm._sigma

    def _restore_iterate(self, algorithm):
        """Restores the iterate to the state it had at the start of the
        iteration.

        :meth:`~cil.optimisation.algorithms.PDHG._pdhg_update` overwrites ``x``
        and ``y`` in place, so before the step can be retaken with smaller step
        sizes both must be put back. Otherwise the retry extrapolates from the
        rejected trial iterate rather than from :math:`x^{k-1}`, and the dual
        variable accumulates over successive trials, which can make the iterate
        overflow before the backtracking loop has had a chance to bring the
        step sizes down.
        """
        algorithm.x.fill(self.x_prev)
        algorithm.y.fill(self.y_old)

    def _calculate_backtracking(self, algorithm):
        """Calculates the backtracking parameter b used to update step sizes in
            the adaptive PDHG algorithm. Returns ------- b : :obj:`float`
                Backtracking parameter used to update step sizes in the
                adaptive PDHG algorithm.
        """

        algorithm.x.sapyb(1.0, algorithm.x_old, -1.0, out=self.x_resid)
        x_change_norm = self.x_resid.norm()
        algorithm.y.sapyb(1.0, self.y_old, -1.0, out=self.y_resid)
        y_change_norm = self.y_resid.norm()
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        cross_term = np.abs(4*algorithm._sigma*algorithm._tau *
                            self.y_resid.dot(algorithm.y_tmp))
        b = self.c*algorithm._sigma*x_change_norm**2 + \
            self.c*algorithm._tau*y_change_norm**2 - cross_term
        log.debug('Backtracking value = {}'.format(b))
        return b

    def _calculate_pnorm_dnorm(self, algorithm):
        """Calculates the primal and dual norms used in the rebalancing step of
        the adaptive PDHG algorithm.
        """
        algorithm.operator.adjoint(self.y_resid, out=algorithm.x_tmp)
        algorithm.operator.direct(self.x_resid, out=algorithm.y_tmp)
        self.x_resid.sapyb((1/algorithm._tau),
                           algorithm.x_tmp, -1.0, out=algorithm.x_tmp)
        self.y_resid.sapyb((1/algorithm._sigma),
                           algorithm.y_tmp, -1.0, out=algorithm.y_tmp)
        self.p_norm = algorithm.x_tmp.norm()
        self.d_norm = algorithm.operator.norm()*algorithm.y_tmp.norm()
