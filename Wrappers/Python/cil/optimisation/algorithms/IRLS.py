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

from cil.optimisation.algorithms import LSQR, CGLS, Algorithm
from cil.optimisation.functions import L1Norm
from cil.optimisation.utilities.callbacks import (Callback, CGLSEarlyStopping,
                                                  InnerCallback,
                                                  IRLSEarlyStopping,
                                                  OuterCallback)
from typing import Union, List, Optional
from tqdm.auto import tqdm

import numpy as np
import logging
import sys
import warnings

log = logging.getLogger(__name__)


class IRLS(Algorithm):
    r"""
    Iteratively Reweighted Least Squares (IRLS) algorithm for solving L1-regularised problems.

    This outer algorithm manages an inner solver (e.g., LSQR or CGLS), iteratively 
    updating a diagonal weight matrix to approximate the L1 norm

    ||u||_1 ~ \sum_i w_i |u_i|^2

    .. math::
        w_k = (|u_{k-1}|^2 + \tau_k^2)^{-1/4}

    where :math:`\tau_k` is a smoothing parameter that is reduced over iterations [1] to improve convergence.
    
    Allowing the inner solver to solve
    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|Lx\|_1

    by iteratively solving a series of weighted L2 problems

    .. math::
        \min_x \|Ax - b\|_2^2 + \alpha^2 \|W_k Lx\|_2^2

    Where :math:`W_k` is a diagonal weight matrix that is updated by the IRLS algorithm.

    Choosing an inner solver
    ------------------------
    Either :class:`LSQR` or :class:`CGLS` will do, but they behave differently
    as the reweighting proceeds, and the differences are not cosmetic.

    * **Conditioning.** LSQR works with the condition number
      of :math:`K`; CGLS works with its square, enough in float32 to diverge,
      so IRLS attaches a :class:`CGLSEarlyStopping` guard to a CGLS inner
      solver by default. Override with ``inner_callbacks``.

    * **Warm starts.** Standard-form LSQR penalises the step rather than the
      solution, so it cannot warm start; CGLS warm starts in either form.
      ``form='auto'`` accounts for this, and attaching IRLS to an
      auto-resolved standard-form LSQR rebuilds it in block form. Only an
      *explicit* ``form='standard'`` is left alone and IRLS warns.

    Stopping the outer loop
    -----------------------
    By default the outer loop runs exactly the number of iterations passed to
    :meth:`run`. Passing ``tol`` instead attaches an :class:`IRLSEarlyStopping`
    callback, which terminates once the relative change between successive
    outer iterates falls below it. The recorded loss is the actual L1
    objective ||Au-b||^2 + alpha^2 ||Lu||_1, at the cost of one application
    of :math:`A` and of :math:`L` every ``update_objective_interval`` outer
    iterations; the inner solver's reweighted loss remains available through
    ``inner_solver.loss``.

    References
    ----------
    .. [1] R. Chartrand and Wotao Yin, "Iteratively reweighted algorithms for
       compressive sensing," 2008 IEEE ICASSP, Las Vegas, NV, USA, 2008.
    """

    def __init__(
        self,
        inner_solver: Union[LSQR, CGLS],
        tau: float = 1.0,
        tau_factor: float = 0.1,
        tau_min: float = 1e-8,
        max_inner_iteration: int = 20,
        reset_state: bool = False,
        inner_callbacks: Optional[List[Callback]] = None,
        tol: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.inner_solver = inner_solver
        self.max_inner_iteration = max_inner_iteration
        self.reset_state = reset_state
        self.tau = tau
        self.tau_factor = tau_factor
        self.tau_min = tau_min
        self.tol = tol

    
        if inner_callbacks is None:
            inner_callbacks = ([CGLSEarlyStopping()]
                               if isinstance(inner_solver, CGLS) else [])
        self.inner_callbacks = list(inner_callbacks)

        # Check that the inner solver's operator exposes the IRLS weight surface
        if not hasattr(self.inner_solver, "operator") or not all(
            hasattr(self.inner_solver.operator, name)
            for name in ("weights", "enable_weights")
        ):
            raise ValueError(
                "The inner solver's operator must expose the IRLS weight "
                "surface: a 'weights' property and 'enable_weights()'. "
                "Build it with create_tikhonov_operator, or match that "
                "interface.")

        # Settle the form before the weights, so they belong to the final operator.
        self._require_warm_startable_inner_solver()

        # No-op if the caller passed weighted=True.
        if self.inner_solver.weights is None:
            log.info("Allocating IRLS weights. Construct the inner solver with "
                     "weighted=True to keep every allocation inside set_up.")
            self.inner_solver.enable_weights()

        # Solution buffer for standard form's (WL)^-1 mapping; block form uses the live iterate.
        self.tmp_solution = (
            self.inner_solver.solution_geometry().allocate(0)
            if self.inner_solver.standard_form else None)

        self.configured = True

    def _require_warm_startable_inner_solver(self):
        """
        Rebuild an auto-resolved standard-form inner solver in block form, so
        it can warm start. An explicit ``form='standard'``, or a solver with
        no way back, is left alone: IRLS warns and falls back to
        ``reset_state=True``.
        """
        if self.inner_solver.supports_warm_start or self.reset_state:
            return

        rebuild = getattr(self.inner_solver, 'rebuild_in_block_form', None)
        if rebuild is not None and rebuild():
            return

        warnings.warn(
            "{} in standard form cannot warm start: it damps the step "
            "rather than the solution, so resuming from the previous outer "
            "iterate does not solve the regularised problem. Forcing "
            "reset_state=True. Build the inner solver with form='block' to "
            "warm start.".format(type(self.inner_solver).__name__),
            UserWarning, stacklevel=3)
        self.reset_state = True

    def run(self, iterations=None, callbacks: Optional[List[Callback]] = None, verbose: int = 1):
        """
        Run the outer loop with one tqdm bar per loop: "Outer Loop" here,
        "Inner Loop" per inner solve. ``callbacks`` run in addition;
        ``verbose=0`` silences both bars.
        """
        if iterations is None:
            raise ValueError("`run()` missing number of `iterations`")

        callbacks = [] if callbacks is None else list(callbacks)

        # `tol` is opt-in; a caller-supplied IRLSEarlyStopping takes precedence.
        if self.tol is not None and not any(
                isinstance(cb, IRLSEarlyStopping) for cb in callbacks):
            callbacks.append(IRLSEarlyStopping(epsilon=self.tol,
                                               verbose=verbose))

        # verbose=0 must silence the bars too.
        self._quiet = verbose == 0

        with tqdm(
            total=iterations,
            desc="Outer Loop",
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=self._quiet,
        ) as outer_pbar:
            outer_cb = OuterCallback(outer_pbar)
            callbacks.append(outer_cb)

            # verbose=0 hides the base class's own logging; the bars report instead.
            super().run(iterations, callbacks=callbacks, verbose=0)

    def update(self):
        """Perform a single outer IRLS iteration."""

        # Get the solution from the inner solver
        solution = self.inner_solver.get_output(out=self.tmp_solution)

        if not self.reset_state:
            self.inner_solver.initial = solution

        # Calculate and inject new L1 weights based on that solution
        self._update_weights(solution)

        # Reset the inner solver
        self.inner_solver.initialise_variables()

        # Inner Loop: Run the Krylov solver with a nested tqdm progress bar
        with tqdm(
            total=self.max_inner_iteration,
            desc="Inner Loop",
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            disable=getattr(self, '_quiet', False),
        ) as inner_pbar:
            inner_cb = InnerCallback(inner_pbar)
            self.inner_solver.run(
                self.max_inner_iteration,
                callbacks=[inner_cb] + self.inner_callbacks,
                verbose=0,
            )

    def _update_weights(self, solution):
        """Update the diagonal L1 weights in-place from the previous outer
        solution. Allocates nothing."""
        op = self.inner_solver.operator

        struct_direct = getattr(op, "struct_direct", None)
        if struct_direct is not None:
            # Map to structure space (Lu), where the weights live.
            struct_direct(solution, out=op.weights)
        else:
            op.weights.fill(solution)

        # weights = (|weights|^2 + tau^2)^(-1/4)
        op.weights.power(2, out=op.weights)
        op.weights.add(self.tau**2, out=op.weights)
        op.weights.power(-0.25, out=op.weights)

        self._adapt_tau()

    def update_objective(self):
        """
        Record the L1 objective ||Au-b||^2 + alpha^2 ||Lu||_1 at the current
        solution, at the cost of one application of A and of L per entry --
        throttle with ``update_objective_interval``. The inner solver's
        reweighted loss stays available through ``inner_solver.loss``.
        """
        op = self.inner_solver.operator
        u = self.inner_solver.get_output()

        residual = op.operator.direct(u)
        residual.sapyb(1.0, self.inner_solver.data, -1.0, out=residual)
        data_term = residual.squared_norm()

        reg_term = L1Norm()(op.struct_direct(u))
        self.loss.append(data_term
                         + self.inner_solver.regalpha**2 * reg_term)

    def _adapt_tau(self):
        """
        Adapts the smoothing parameter tau.
        Reduces by tau_factor until it hits the tau_min floor.
        """
        self.tau = max(self.tau * self.tau_factor, self.tau_min)

    def get_output(self, out=None):
        """Returns the final physical solution from the inner solver."""
        return self.inner_solver.get_output(out=out)