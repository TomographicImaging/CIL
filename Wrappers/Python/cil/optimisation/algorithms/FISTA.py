# -*- coding: utf-8 -*-
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
from cil.optimisation.functions import ZeroFunction
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
        Differentiable function. If `None` is passed, the algorithm will use the ZeroFunction.
    g : Function or `None`
        Convex function with *simple* proximal operator. If `None` is passed, the algorithm will use the ZeroFunction.
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of ISTA.
                The default :code:`step_size` is :math:`\frac{1}{L}` or 1 if `f=None`.
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.

    Note
    -----
    If the function `g` is set to `None` or to the `ZeroFunction` then the ISTA algorithm is equivalent to Gradient Descent. 
    
    If the function `f` is set to `None` or to the `ZeroFunction` then the ISTA algorithm is equivalent to a Proximal Point Algorithm. 

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

    def _provable_convergence_condition(self):
        return self.step_size <= 0.99*2.0/self.f.L

    @property
    def step_size(self):        
       return self._step_size

    # Set default step size
    def set_step_size(self, step_size):
        """ Set default step size.
        """
    
        if step_size is None:
            if isinstance(self.f, ZeroFunction):
                self._step_size = 1
                
            elif isinstance(self.f.L, Number):
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

        logging.info("{} setting up".format(self.__class__.__name__, ))        

        # set up ISTA      
        self.initial = initial
        self.x_old = initial.copy()
        self.x = initial.copy()    
        
        if f is None:
            f = ZeroFunction()
            
            if g is None: 
                raise ValueError('You set both f and g to be None and thus the iterative method will not update and will remain fixed at the initial value.')
 
                
        self.f = f
        
        if g is None:
            g = ZeroFunction()
            
        self.g = g

        # set step_size
        self.set_step_size(step_size=step_size)
        self.configured = True  

        logging.info("{} configured".format(self.__class__.__name__, ))
              

    def update(self):

        r"""Performs a single iteration of ISTA

        .. math:: x_{k+1} = \mathrm{prox}_{\alpha g}(x_{k} - \alpha\nabla f(x_{k}))

        """

        # gradient step
        self.f.gradient(self.x_old, out=self.x)
        self.x_old.sapyb(1., self.x, -self.step_size, out=self.x_old)

        # proximal step
        self.g.proximal(self.x_old, self.step_size, out=self.x)

    def _update_previous_solution(self):  
        """ Swaps the references to current and previous solution based on the :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """        
        tmp = self.x_old
        self.x_old = self.x
        self.x = tmp

    def get_output(self):
        " Returns the current solution. "
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
        Differentiable function.  If `None` is passed, the algorithm will use the ZeroFunction.
    g : Function or `None`
        Convex function with *simple* proximal operator. If `None` is passed, the algorithm will use the ZeroFunction.
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of FISTA.
                The default :code:`step_size` is :math:`\frac{1}{L}` or 1 if `f=None`.
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.

    Note
    -----
    If the function `g` is set to `None` or to the `ZeroFunction` then the FISTA algorithm is equivalent to Accelerated Gradient Descent by Nesterov (:cite:`nesterov2003introductory` algorithm 2.2.9).

    If the function `f` is set to `None` or to the `ZeroFunction` then the FISTA algorithm is equivalent to Guler's First Accelerated Proximal Point Method  (:cite:`guler1992new` sec 2).

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
            
            if isinstance(self.f, ZeroFunction):
                self._step_size = 1
                
            elif isinstance(self.f.L, Number):
                self._step_size = 1./self.f.L
                
            else:
                raise ValueError("Function f is not differentiable")
            
        else:
            self._step_size = step_size

    def _provable_convergence_condition(self):
        return self.step_size <= 1./self.f.L

    def __init__(self, initial, f, g, step_size = None, **kwargs):

        self.y = initial.copy()
        self.t = 1
        super(FISTA, self).__init__(initial=initial, f=f, g=g, step_size=step_size, **kwargs)
              
    def update(self):
        
        r"""Performs a single iteration of FISTA

        .. math::

            \begin{cases}
                x_{k+1} = \mathrm{prox}_{\alpha g}(y_{k} - \alpha\nabla f(y_{k}))\\
                t_{k+1} = \frac{1+\sqrt{1+ 4t_{k}^{2}}}{2}\\
                y_{k+1} = x_{k} + \frac{t_{k}-1}{t_{k-1}}(x_{k} - x_{k-1})
            \end{cases}

        """        

        self.t_old = self.t

        self.f.gradient(self.y, out=self.x)
        
        self.y.sapyb(1., self.x, -self.step_size, out=self.y)
        
        self.g.proximal(self.y, self.step_size, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.x.subtract(self.x_old, out=self.y)
        self.y.sapyb(((self.t_old-1)/self.t), self.x, 1.0, out=self.y) 
          

if __name__ == "__main__":

    from cil.optimisation.functions import L2NormSquared
    from cil.optimisation.algorithms import GD
    from cil.framework import ImageGeometry
    f = L2NormSquared()
    g = L2NormSquared()
    ig = ImageGeometry(3,4,4)
    initial = ig.allocate()
    fista = FISTA(initial, f, g, step_size = 1443432)
    print(fista.is_provably_convergent())

    gd = GD(initial=initial, objective = f, step_size = 1023123)
    print(gd.is_provably_convergent())

