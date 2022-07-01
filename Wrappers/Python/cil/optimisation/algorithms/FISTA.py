# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cil.optimisation.algorithms import Algorithm
import numpy
import warnings
import logging
from numbers import Number


class ISTA(Algorithm):

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

    >>> f = LeastSquares(A, b=b, c=0.5)
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
    def provable_convergence_condition(self):
        return self.step_size <= 0.99*2.0/self.f.L

    @property
    def step_size(self):        
       return self._step_size

    # Set default step size
    def set_step_size(self, step_size):
        """ Set default step size.
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self._step_size = 0.99*2.0/self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self._step_size = step_size            
        
    def __init__(self, initial, f, g, step_size = None, **kwargs):

        super(ISTA, self).__init__(**kwargs)
        self._step_size = None
        self.set_up(initial=initial, f=f, g=g, step_size=step_size, **kwargs)

    def set_up(self, initial, f, g, step_size, **kwargs):
        """ Set up of the algorithm
        """

        # set up ISTA      
        self.initial = initial
        self.x_old = initial.copy()
        self.x = initial.copy()           
        self.f = f
        self.g = g

        # set step_size
        self.set_step_size(step_size=step_size)

        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        self.configured = True  

        logging.info("{} configured".format(self.__class__.__name__, ))
              

    def update(self):

        r"""Performs a single iteration of ISTA

        .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

        """

        # gradient step
        self.x_old.sapyb(1., self.f.gradient(self.x_old), -self.step_size, out=self.x)

        # proximal step
        self.x = self.g.proximal(self.x, self.step_size)

    def update_previous_solution(self):  
        """ Swap the pointers to current and previous solution based on the :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """        
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def get_output(self):
        " Overrides the base method :func:`~Algorithm.get_output` of the base class :class:`Algorithm`."
        return self.x_old
        
    def update_objective(self):
        """ Updates the objective

        .. math:: f(x) + g(x)

        """
        self.loss.append( self.f(self.x_old) + self.g(self.x_old) )


class FISTA(ISTA):

    r"""Fast Iterative Shrinkage-Thresholding Algorithm, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.

    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA)

    .. math::

        \begin{cases}
            y_{k} = x_{k} - \alpha\nabla f(x_{k})  \\
            x_{k+1} = \mathrm{prox}_{\alpha g}(y_{k})\\
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


    >>> f = LeastSquares(A, b=b, c=0.5)
    >>> g = ZeroFunction()
    >>> ig = Aop.domain
    >>> fista = FISTA(initial = ig.allocate(), f = f, g = g, max_iteration=10)
    >>> fista.run()

    See also
    --------
    :class:`.FISTA`
    :class:`.GD`

    """

    def set_step_size(self, step_size):

        """Set the default step size
        """
        if step_size is None:
            if isinstance(self.f.L, Number):
                self._step_size = 1./self.f.L
            else:
                raise ValueError("Function f is not differentiable")
        else:
            self._step_size = step_size

    @property
    def provable_convergence_condition(self):
        return self.step_size <= 1./self.f.L

    def __init__(self, initial, f, g, step_size = None, **kwargs):

        self.y = initial.copy()
        self.t = 1
        super(FISTA, self).__init__(initial=initial, f=f, g=g, step_size=step_size, **kwargs)


    def update(self):
        
        r"""Performs a single iteration of FISTA

        .. math::

            \begin{cases}
                y_{k} = x_{k} - \alpha\nabla f(x_{k})  \\
                x_{k+1} = \mathrm{prox}_{\alpha g}(y_{k})\\
                t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
                y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k-1}}(x_{k} - x_{k-1})
            \end{cases}

        """        

        self.t_old = self.t
        
        self.y.sapyb(1., self.f.gradient(self.x_old), -self.step_size, out=self.y)
        
        self.g.proximal(self.y, self.step_size, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.x.subtract(self.x_old, out=self.y)
        self.y.sapyb(((self.t_old-1)/self.t), self.x, 1.0, out=self.y) 


if __name__ == "__main__":

    from cil.optimisation.functions import L2NormSquared
    from cil.framework import ImageGeometry
    import time
    f = L2NormSquared()
    g = L2NormSquared()
    ig = ImageGeometry(3,4)

    initial = ig.allocate()
    t1 = time.time()
    ista = ISTA(initial = initial,  f = f, g = g, step_size=0.1)
    t2 = time.time()
    print(t2-t1)
    
    t3 = time.time()
    print(ista.provable_convergence_condition)
    t4 = time.time()
    print(t4-t3)    