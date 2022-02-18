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
from cil.optimisation.algorithms import Algorithm
import warnings
from numbers import Number

class FISTA(Algorithm):
    
    r""" Fast Iterative Shrinkage-Thresholding Algorithm (FISTA), see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`. 
    
    Problem : 
    
    .. math::
    
      \min_{x} f(x) + g(x)
    
    |
    
    Parameters :
        
      :param initial: Initial guess ( Default initial = 0)
      :param f: Differentiable function
      :param g: Convex function with " simple " proximal operator


    Reference:
      
        Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. 
        SIAM journal on imaging sciences,2(1), pp.183-202.
    """
    
    
    def __init__(self, initial, f, g, step_size = None, **kwargs):
        
        """ FISTA algorithm 
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up
        
        Optional parameters:

        :param initial: Initial guess ( Default initial = 0)
        :param f: Differentiable function
        :param g: Convex function with " simple " proximal operator"""
        
        super(FISTA, self).__init__(**kwargs)
        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
        
        self.set_up(initial=initial, f=f, g=g, step_size=step_size)

    def set_up(self, initial, f, g=ZeroFunction()):
        '''initialisation of the algorithm

        :param initial: Initial guess ( Default initial = 0)
        :param f: Differentiable function
        :param g: Convex function with " simple " proximal operator'''

        print("{} setting up".format(self.__class__.__name__, ))
        
        self.y = initial.copy()
        self.x_old = initial.copy()
        self.x = initial.copy()
        self.u = initial.copy()

        self.f = f
        self.g = g
        if f.L is None:
            raise ValueError('Error: Fidelity Function\'s Lipschitz constant is set to None')
        self.invL = 1/f.L
        self.t = 1
        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

            
    def update(self):
        self.t_old = self.t
        self.f.gradient(self.y, out=self.u)
        self.u.__imul__( -self.invL )
        self.u.__iadd__( self.y )

        self.g.proximal(self.u, self.invL, out=self.x)
        
        self.t = 0.5*(1 + numpy.sqrt(1 + 4*(self.t_old**2)))
        
        self.x.subtract(self.x_old, out=self.y)
        self.y.axpby(((self.t_old-1)/self.t), 1, self.x, out=self.y)
        
        self.x_old.fill(self.x)

        
    def update_objective(self):
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
                The default :code:`step_size` is :math:`\frac{0.99 * 2}{L}`.    

    **kwargs:
        Keyward arguments used from the base class :class:`.Algorithm`.    
    
        max_iteration : :obj:`int`, optional, default=0
            Maximum number of iterations of ISTA.
        update_objective_interval : :obj:`int`, optional, default=1
            Evaluates objective every ``update_objective_interval``.
             

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

    def check_stepsize(self):
        if isinstance(self.step_size, Number) and self.step_size<=0:
            raise ValueError("Positive step size is required. Got {}".format(self.step_size))

    def check_convergence(self):
        if self.step_size >= 0.99*2/f.L:
            warnings.warn("Convergence criterion of ISTA is not satisfied.")
    
    
    def __init__(self, initial, f, g, step_size = None, **kwargs):
        
        super(ISTA, self).__init__(**kwargs)

        # if kwargs.get('x_init', None) is not None:
        #     if initial is None:
        #         warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
        #            DeprecationWarning, stacklevel=4)
        #         initial = kwargs.get('x_init', None)
        #     else:
        #         raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
        #             .format(self.__class__.__name__))

                
        self.set_up(initial=initial, f=f, g=g, step_size=step_size)

    def set_up(self, initial, f, g, step_size):
        """ Set up of the algorithm
        """

        print("{} setting up".format(self.__class__.__name__, ))
        
        # starting point        
        self.initial = initial

        # step size
        self.step_size = step_size        

        # allocate 
        self.x_old = initial.copy()
        self.x = initial.copy()

        # functions
        self.f = f
        self.g = g

        if f.L is None:
            raise ValueError('Error: Fidelity Function\'s Lipschitz constant is set to None')

        # Check option for step-size            
        if self.step_size is None:
            self.step_size = 0.99 * 2/f.L
        else:
            self.step_size = step_size
        
        #check value of step_size
        self.check_stepsize()

        #check convergence criterion
        self.check_convergence()

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

                    
    def update(self):

        r"""Performs a single iteration of ISTA

        .. math:: x^{k+1} = \mathrm{prox}_{\alpha g}(x^{k} - \alpha\nabla f(x^{k}))
    
        """
        
        # gradient step
        self.f.gradient(self.x_old, out=self.x)
        self.x *= -self.step_size
        self.x += self.x_old
        
        # proximal step
        self.g.proximal(self.x, self.step_size, out=self.x)

        # update
        self.x_old.fill(self.x)
        
    # def update_objective(self):
    #     """Computes the objective :math:`f(x) + g(x)` .
    #     """
    #     self.loss.append( self.f(self.x) + self.g(self.x) )    



if __name__ == "__main__":
    
    # from cil.optimisation.algorithms import ISTA
    import numpy as np
    from cil.framework import VectorData
    from cil.optimisation.operators import MatrixOperator
    from cil.optimisation.functions import LeastSquares, ZeroFunction
    np.random.seed(10)
    n, m = 50, 500
    A = np.random.uniform(0,1, (m, n)).astype('float32') # (numpy array)
    b = (A.dot(np.random.randn(n)) + 0.1*np.random.randn(m)).astype('float32') # (numpy vector)
    Aop = MatrixOperator(A) # (CIL operator)
    bop = VectorData(b) # (CIL VectorData)
    f = LeastSquares(Aop, b=bop, c=0.5)
    g = ZeroFunction()
    ig = Aop.domain
    print(f.L)
    ista = ISTA(initial = ig.allocate(), f = f, g = g, max_iteration=10, line_search = True)     
    # ista.step_size = -4
    # print(ista.step_size)
    print(ista.line_search)
    # ista.run()    



