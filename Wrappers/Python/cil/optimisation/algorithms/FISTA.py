# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi
#   (Collaborative Computational Project in Tomographic Imaging), with
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.algorithms import Algorithm
import numpy
import warnings
from numbers import Number

class FISTA(Algorithm):

    r"""Fast Iterative Shrinkage-Thresholding Algorithm, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.

    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    .. math::

        \begin{cases}
            x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))\\
            t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
            y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k-1}}(x_{k} - x_{k-1})
        \end{cases}

    is used to solve

    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha^{k}`
    is the :code:`step_size` per iteration.


    Parameters
    ----------

    initial : DataContainer
            Starting point of the algorithm
    f : Function
        Differentiable function
    g : Function
        Convex function with *simple* proximal operator
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of FISTA.
                The default :code:`step_size` is :math:`\frac{1}{L}`.



    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.



    Examples
    --------

    .. math:: \underset{x}{\mathrm{argmin}}\|A x - b\|^{2}_{2}


    >>> from cil.optimisation.algorithms import FISTA
    >>> import numpy as np
    >>> from cil.framework import VectorData
    >>> from cil.optimisation.operators import MatrixOperator
    >>> from cil.optimisation.functions import LeastSquares, ZeroFunction
    >>> np.random.seed(10)
    >>> n, m = 50, 500
    >>> A = np.random.uniform(0,1, (m, n)).astype('float32') # (numpy array)
    >>> b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32') # (numpy vector)
    >>> Aop = MatrixOperator(A) # (CIL operator)
    >>> bop = VectorData(b) # (CIL VectorData)
    >>> f = LeastSquares(Aop, b=bop, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> fista = FISTA(initial = ig.allocate(), f = f, g = g, max_iteration=10)
    >>> fista.run()

    See also
    --------
    :class:`.FISTA`
    :class:`.GD`

    """

    @property
    def step_size(self):
        return self._step_size

    @step_size.setter
    def step_size(self, val):
        if isinstance(val, Number):
            if val<=0:
                raise ValueError("Positive step size is required. Got {}".format(val))
            self._step_size = val
        else:
            raise ValueError("Step size is not a number. Got {}".format(val))

    def _set_step_size(self, step_size):

        """Set the default step size
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self.step_size = 1./self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self.step_size = step_size

    @property
    def convergence_criterion(self):
        return self.step_size > 1./self.f.L

    def _check_convergence_criterion(self):
        """Check convergence criterion
        """
        if isinstance(self.f.L, Number):
            if self.convergence_criterion:
                warnings.warn("Convergence criterion is not satisfied.")
                return False
            return True
        else:
            raise ValueError("Function f is not differentiable")

    def __init__(self, initial, f, g, step_size = None, **kwargs):

        super(FISTA, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))

        self._step_size = None

        # set up FISTA
        self.set_up(initial=initial, f=f, g=g, step_size=step_size, **kwargs)

    def set_up(self, initial, f, g, step_size, **kwargs):
        """ Set up of the algorithm
        """

        self.initial = initial
        self.f = f
        self.g = g

        # set step_size
        self._set_step_size(step_size=step_size)

        # check convergence criterion for FISTA is satisfied
        if kwargs.get('check_convergence_criterion', True):
            self._check_convergence_criterion()

        print("{} setting up".format(self.__class__.__name__, ))

        self.y = initial.copy()
        self.x_old = initial.copy()
        self.x = initial.copy()
        self.u = initial.copy()

        self.t = 1
        self.configured = True

        print("{} configured".format(self.__class__.__name__, ))


    def update(self):

        r"""Performs a single iteration of FISTA

        .. math::

            \begin{cases}
                x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))\\
                t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
                y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k-1}}(x_{k} - x_{k-1})
            \end{cases}

        """

        self.t_old = self.t
        self.f.gradient(self.y, out=self.u)
        self.u *= -self.step_size
        self.u += self.y

        self.g.proximal(self.u, self.step_size, out=self.x)

        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))

        self.x.subtract(self.x_old, out=self.y)
        self.y.axpby(((self.t_old-1)/self.t), 1, self.x, out=self.y)

        self.x_old.fill(self.x)


    def update_objective(self):
        """ Updates the objective

        .. math:: f(x) + g(x)

        """
        self.loss.append( self.f(self.x) + self.g(self.x) )

class ISTA(FISTA):

    r"""Iterative Shrinkage-Thresholding Algorithm, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.

    Iterative Shrinkage-Thresholding Algorithm (ISTA)

    .. math:: x^{k+1} = \mathrm{prox}_{\alpha^{k} g}(x^{k} - \alpha^{k}\nabla f(x^{k}))

    is used to solve

    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`\alpha^{k}`
    is the :code:`step_size` per iteration.

    Note
    ----

    For a constant step size, i.e., :math:`a^{k}=a` for :math:`k\geq1`, convergence of ISTA
    is guaranteed if

    .. math:: \alpha\in(0, \frac{2}{L}),

    where :math:`L` is the Lipschitz constant of :math:`f`, see :cite:`CombettesValerie`.

    Parameters
    ----------

    initial : DataContainer
              Initial guess of ISTA.
    f : Function
        Differentiable function
    g : Function
        Convex function with *simple* proximal operator
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of ISTA.
                The default :code:`step_size` is :math:`\frac{0.99 * 2}{L}.`



    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.



    Examples
    --------

    .. math:: \underset{x}{\mathrm{argmin}}\|A x - b\|^{2}_{2}


    >>> from cil.optimisation.algorithms import ISTA
    >>> import numpy as np
    >>> from cil.framework import VectorData
    >>> from cil.optimisation.operators import MatrixOperator
    >>> from cil.optimisation.functions import LeastSquares, ZeroFunction
    >>> np.random.seed(10)
    >>> n, m = 50, 500
    >>> A = np.random.uniform(0,1, (m, n)).astype('float32') # (numpy array)
    >>> b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32') # (numpy vector)
    >>> Aop = MatrixOperator(A) # (CIL operator)
    >>> bop = VectorData(b) # (CIL VectorData)
    >>> f = LeastSquares(Aop, b=bop, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> ista = ISTA(initial = ig.allocate(), f = f, g = g, max_iteration=10)
    >>> ista.run()


    See also
    --------

    :class:`.FISTA`
    :class:`.GD`


    """

    @property
    def convergence_criterion(self):
        return self.step_size > 0.99*2.0/self.f.L

    def __init__(self, initial, f, g, step_size = None, **kwargs):

        super(ISTA, self).__init__(initial=initial, f=f, g=g, step_size=step_size, **kwargs)

    def _set_step_size(self, step_size):
        """ Set default step size.
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self.step_size = 0.99*2.0/self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self.step_size = step_size

    def update(self):

        r"""Performs a single iteration of ISTA

        .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

        """

        # gradient step
        self.f.gradient(self.x_old, out=self.x)
        self.x *= -self.step_size
        self.x += self.x_old

        # proximal step
        self.g.proximal(self.x, self.step_size, out=self.x)

        # update
        self.x_old.fill(self.x)

