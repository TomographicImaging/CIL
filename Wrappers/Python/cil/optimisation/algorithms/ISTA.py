from cil.optimisation.algorithms import Algorithm
import warnings

class ISTA(Algorithm):
    
    r"""Iterative Shrinkage-Thresholding Algorithm, see :cite:`BeckTeboulle_a`, :cite:`BeckTeboulle_b`.
    
    Iterative Shrinkage-Thresholding Algorithm (ISTA) 
    
    .. math:: x^{k+1} = \mathrm{prox}_{L g}(x^{k} - \frac{1}{L}\nabla f(x^{k}))
    
    is used to solve 
    
    .. math:: \min_{x} f(x) + g(x)

    where :math:`f` is differentiable, :math:`g` has a *simple* proximal operator and :math:`L`
    is the Lipshcitz constant of the function :math:`f`.

    Parameters
    ----------

    initial : DataContainer
              Initial guess of ISTA.
    f : Function
        Differentiable function 
    g : Function
        Convex function with *simple* proximal operator
    step_size : positive :obj:`float`, default = None
                Step size for the gradient step of ISTA


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
      
    """
    
    
    def __init__(self, initial, f, g, step_size = None, **kwargs):
        
        super(ISTA, self).__init__(**kwargs)

        if kwargs.get('x_init', None) is not None:
            if initial is None:
                warnings.warn('The use of the x_init parameter is deprecated and will be removed in following version. Use initial instead',
                   DeprecationWarning, stacklevel=4)
                initial = kwargs.get('x_init', None)
            else:
                raise ValueError('{} received both initial and the deprecated x_init parameter. It is not clear which one we should use.'\
                    .format(self.__class__.__name__))
                
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
        
        self.t = 1

        self.configured = True
        print("{} configured".format(self.__class__.__name__, ))

                    
    def update(self):

        r"""Performs a single iteration of ISTA
        """
        
        # gradient step
        self.f.gradient(self.x_old, out=self.x)
        self.x *= -self.step_size
        self.x += self.x_old
        
        # proximal step
        self.g.proximal(self.x, self.step_size, out=self.x)

        # update
        self.x_old.fill(self.x)
        
    def update_objective(self):
        """Computes the objective 
        """
        self.loss.append( self.f(self.x) + self.g(self.x) )    
