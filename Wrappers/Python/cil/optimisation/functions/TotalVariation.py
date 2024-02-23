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
# Claire Delplancke (University of Bath)


from cil.optimisation.functions import Function, IndicatorBox, MixedL21Norm, MixedL11Norm
from cil.optimisation.operators import GradientOperator
import numpy as np
from numbers import Number
import warnings
import logging


class TotalVariation(Function):

    r""" Total variation Function

    .. math:: \mathrm{TV}(u) := \|\nabla u\|_{2,1} = \sum \|\nabla u\|_{2},\, (\mbox{isotropic})

    .. math:: \mathrm{TV}(u) := \|\nabla u\|_{1,1} = \sum \|\nabla u\|_{1}\, (\mbox{anisotropic})

    Notes
    -----

    The :code:`TotalVariation` (TV) :code:`Function` acts as a composite function, i.e.,
    the composition of the :class:`.MixedL21Norm` function and the :class:`.GradientOperator` operator,

    .. math:: f(u) = \|u\|_{2,1}, \Rightarrow (f\circ\nabla)(u) = f(\nabla x) = \mathrm{TV}(u)

    In that case, the proximal operator of TV does not have an exact solution and we use an iterative 
    algorithm to solve:

    .. math:: \mathrm{prox}_{\tau \mathrm{TV}}(b) := \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TV}(u)

    The algorithm used for the proximal operator of TV is the Fast Gradient Projection algorithm (or FISTA)
    applied to the _dual problem_ of the above problem, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`, :cite:`Zhu2010`.

    See also "Multicontrast MRI Reconstruction with Structure-Guided Total Variation", Ehrhardt, Betcke, 2016.


    Parameters
    ----------

    max_iteration : :obj:`int`, default = 5
        Maximum number of iterations for the FGP algorithm to solve to solve the dual problem 
        of the Total Variation Denoising problem (ROF). If warm_start=False, this should be around 100,
        or larger, with a set tolerance. 
    tolerance : :obj:`float`, default = None
        Stopping criterion for the FGP algorithm used to to solve the dual problem 
        of the Total Variation Denoising problem (ROF). If the difference between iterates in the FGP algorithm is less than the tolerance
        the iterations end before the max_iteration number. 

        .. math:: \|x^{k+1} - x^{k}\|_{2} < \mathrm{tolerance}

    correlation : :obj:`str`, default = `Space`
        Correlation between `Space` and/or `SpaceChannels` for the :class:`.GradientOperator`.
    backend :  :obj:`str`, default = `c`      
        Backend to compute the :class:`.GradientOperator`
    lower : :obj:`'float`, default = None
        A constraint is enforced using the :class:`.IndicatorBox` function, e.g., :code:`IndicatorBox(lower, upper)`.
    upper : :obj:`'float`, default = None
        A constraint is enforced using the :class:`.IndicatorBox` function, e.g., :code:`IndicatorBox(lower, upper)`.  
    isotropic : :obj:`boolean`, default = True
        Use either isotropic or anisotropic definition of TV.

        .. math:: |x|_{2} = \sqrt{x_{1}^{2} + x_{2}^{2}},\, (\mbox{isotropic})

        .. math:: |x|_{1} = |x_{1}| + |x_{2}|\, (\mbox{anisotropic})

    split : :obj:`boolean`, default = False
        Splits the Gradient into spatial gradient and spectral or temporal gradient for multichannel data.

    info : :obj:`boolean`, default = False
        Information is printed for the stopping criterion of the FGP algorithm used to solve the dual problem
        of the Total Variation Denoising problem (ROF).

    strong_convexity_constant : :obj:`float`, default = 0
        A strongly convex term weighted by the :code:`strong_convexity_constant` (:math:`\gamma`) parameter is added to the Total variation. 
        Now the :code:`TotalVariation` function is :math:`\gamma` - strongly convex and the proximal operator is

        .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TV}(u) + \frac{\gamma}{2}\|u\|^{2} \Leftrightarrow

        .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2\frac{\tau}{1+\gamma\tau}}\|u - \frac{b}{1+\gamma\tau}\|^{2} + \mathrm{TV}(u) 

    warm_start : :obj:`boolean`, default = True
        If set to true, the FGP algorithm used to solve the dual problem of the Total Variation Denoising problem (ROF) is initiated by the final value from the previous iteration and not at zero. 
        This allows the max_iteration value to be reduced to 5-10 iterations. 


    Note
    ----

    With warm_start set to the default, True, the TV function will keep in memory the range of the gradient of the image to be denoised, i.e. N times the dimensionality of the image. This increases the memory requirements. 
    However, during the evaluation of `proximal` the memory requirements will be unchanged as the same amount of memory will need to be allocated and deallocated. 

    Note
    ----

    In the case where the Total variation becomes a :math:`\gamma` - strongly convex function, i.e.,

    .. math:: \mathrm{TV}(u) + \frac{\gamma}{2}\|u\|^{2}

    :math:`\gamma` should be relatively small, so as the second term above will not act as an additional regulariser.
    For more information, see :cite:`Rasch2020`, :cite:`CP2011`.




    Examples
    --------

    .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + \alpha\|\nabla u\|_{2,1}

    >>> alpha = 2.0
    >>> TV = TotalVariation()
    >>> sol = TV.proximal(b, tau = alpha)

    Examples
    --------

    .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + \alpha\|\nabla u\|_{1,1} + \mathbb{I}_{C}(u)

    where :math:`C = \{1.0\leq u\leq 2.0\}`.

    >>> alpha = 2.0
    >>> TV = TotalVariation(isotropic=False, lower=1.0, upper=2.0)
    >>> sol = TV.proximal(b, tau = alpha)    


    Examples
    --------

    .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + (\alpha\|\nabla u\|_{2,1} + \frac{\gamma}{2}\|u\|^{2})

    >>> alpha = 2.0
    >>> gamma = 1e-3
    >>> TV = alpha * TotalVariation(isotropic=False, strong_convexity_constant=gamma)
    >>> sol = TV.proximal(b, tau = 1.0)    

    """

    def __init__(self,
                 max_iteration=10,
                 tolerance=None,
                 correlation="Space",
                 backend="c",
                 lower=None,
                 upper=None,
                 isotropic=True,
                 split=False,
                 info=False,
                 strong_convexity_constant=0,
                 warm_start=True):

        super(TotalVariation, self).__init__(L=None)

        # Regularising parameter = alpha
        self.regularisation_parameter = 1.

        self.iterations = max_iteration

        self.tolerance = tolerance

        # Total variation correlation (isotropic=Default)
        self.isotropic = isotropic

        # correlation space or spacechannels
        self.correlation = correlation
        self.backend = backend

        # Define orthogonal projection onto the convex set C
        if lower is None:
            lower = -np.inf
        if upper is None:
            upper = np.inf
        self.lower = lower
        self.upper = upper
        self.projection_C = IndicatorBox(lower, upper).proximal

        # Setup GradientOperator as None. This is to avoid domain argument in the __init__
        self._gradient = None
        self._domain = None

        self.info = info
        if self.info:
            warnings.warn(" `info` is deprecate. Please use logging instead.")

        # splitting Gradient
        self.split = split

        # For the warm_start functionality
        self.warm_start = warm_start
        self._p2 = None

        # Strong convexity for TV
        self.strong_convexity_constant = strong_convexity_constant

        # Define Total variation norm
        if self.isotropic:
            self.func = MixedL21Norm()
        else:
            self.func = MixedL11Norm()

    def _get_p2(self):
        r"""The initial value for the dual in the proximal calculation - allocated to zero in the case of warm_start=False
          or initialised as the last iterate seen in the proximal calculation in the case warm_start=True ."""

        if self._p2 is None:
            return self.gradient.range_geometry().allocate(0)
        else:
            return self._p2

    @property
    def regularisation_parameter(self):
        return self._regularisation_parameter

    @regularisation_parameter.setter
    def regularisation_parameter(self, value):
        if not isinstance(value, Number):
            raise TypeError(
                "regularisation_parameter: expected a number, got {}".format(type(value)))
        self._regularisation_parameter = value

    def __call__(self, x):
        r""" Returns the value of the TotalVariation function at :code:`x` ."""

        try:
            self._domain = x.geometry
        except:
            self._domain = x

        # Compute Lipschitz constant provided that domain is not None.
        # Lipschitz constant dependes on the GradientOperator, which is configured only if domain is not None
        if self._L is None:
            self.calculate_Lipschitz()

        if self.strong_convexity_constant > 0:
            strongly_convex_term = (
                self.strong_convexity_constant/2)*x.squared_norm()
        else:
            strongly_convex_term = 0

        return self.regularisation_parameter * self.func(self.gradient.direct(x)) + strongly_convex_term

    def proximal(self, x, tau, out=None):
        r""" Returns the proximal operator of the TotalVariation function at :code:`x` ."""

        if self.strong_convexity_constant > 0:

            strongly_convex_factor = (1 + tau * self.strong_convexity_constant)
            x /= strongly_convex_factor
            tau /= strongly_convex_factor

        if out is None:
            solution = self._fista_on_dual_rof(x, tau)
        else:
            self._fista_on_dual_rof(x, tau, out=out)

        if self.strong_convexity_constant > 0:
            x *= strongly_convex_factor
            tau *= strongly_convex_factor

        if out is None:
            return solution

    def _fista_on_dual_rof(self, x, tau, out=None):
        r""" Runs the Fast Gradient Projection (FGP) algorithm to solve the dual problem 
        of the Total Variation Denoising problem (ROF).

        .. math: \max_{\|y\|_{\infty}<=1.} \frac{1}{2}\|\nabla^{*} y + x \|^{2} - \frac{1}{2}\|x\|^{2}

        """
        try:
            self._domain = x.geometry
        except:
            self._domain = x

        # Compute Lipschitz constant provided that domain is not None.
        # Lipschitz constant depends on the GradientOperator, which is configured only if domain is not None
        if self._L is None:
            self.calculate_Lipschitz()

        # initialise
        t = 1

        # dual variable - its content is overwritten during iterations
        p1 = self.gradient.range_geometry().allocate(None)
        p2 = self._get_p2()
        tmp_q = p2.copy()

        # multiply tau by -1 * regularisation_parameter here so it's not recomputed every iteration
        # when tau is an array this is done inplace so reverted at the end
        if isinstance(tau, Number):
            tau_reg_neg = -self.regularisation_parameter * tau
        else:
            tau_reg_neg = tau
            tau.multiply(-self.regularisation_parameter, out=tau_reg_neg)

        should_return = False
        if out is None:
            should_return = True
            out = self.gradient.domain_geometry().allocate(0)

        for k in range(self.iterations):

            t0 = t
            self.gradient.adjoint(tmp_q, out=out)
            out.sapyb(tau_reg_neg, x, 1.0, out=out)
            self.projection_C(out, tau=None, out=out)

            self.gradient.direct(out, out=p1)

            multip = (-self.L)/tau_reg_neg

            tmp_q.sapyb(1., p1, multip, out=tmp_q)

            if self.tolerance is not None and k % 5 == 0:
                p1 *= multip
                error = p1.norm()
                error /= tmp_q.norm()
                if error < self.tolerance:
                    break

            self.func.proximal_conjugate(tmp_q, 1.0, out=p1)

            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
            p1.subtract(p2, out=tmp_q)
            tmp_q *= (t0-1)/t
            tmp_q += p1

            # switch p1 and p2 references
            tmp = p1
            p1 = p2
            p2 = tmp
        if self.warm_start:
            self._p2 = p2

        if self.info:
            if self.tolerance is not None:
                logging.info(
                    "Stop at {} iterations with tolerance {} .".format(k, error))
            else:
                logging.info("Stop at {} iterations.".format(k))

        # return tau to its original state if it was modified
        if id(tau_reg_neg) == id(tau):
            tau_reg_neg.divide(-self.regularisation_parameter, out=tau)

        if should_return:
            return out

    def convex_conjugate(self, x):
        r""" Returns the value of convex conjugate of the TotalVariation function at :code:`x` ."""
        return 0.0

    def calculate_Lipschitz(self):
        r""" Default value for the Lipschitz constant."""

        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        self._L = (1./self.gradient.norm())**2

    @property
    def gradient(self):
        r""" GradientOperator is created if it is not instantiated yet. The domain of the `_gradient`,
        is created in the `__call__` and `proximal` methods. 

        """
        if self._domain is not None:
            self._gradient = GradientOperator(
                self._domain, correlation=self.correlation, backend=self.backend)
        else:
            raise ValueError(
                " The domain of the TotalVariation is {}. Please use the __call__ or proximal methods first before calling gradient.".format(self._domain))

        return self._gradient

    def __rmul__(self, scalar):
        if not isinstance(scalar, Number):
            raise TypeError(
                "scalar: Expected a number, got {}".format(type(scalar)))
        self.regularisation_parameter *= scalar
        return self
