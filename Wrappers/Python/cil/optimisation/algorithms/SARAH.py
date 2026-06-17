from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import ApproximateGradientSumFunction
from cil.optimisation.utilities import ConstantStepSize
from numbers import Number
import logging

class SARAH(Algorithm):

    r"""SARAH algorithm.

    StochAstic Recursive grAdient algoritHm (SARAH)
    Lam M. Nguyen, Jie Liu, Katya Scheinberg, Martin Takáč 
    Proceedings of the 34th International Conference on Machine Learning, PMLR 70:2613-2621, 2017. 
    https://proceedings.mlr.press/v70/nguyen17b/nguyen17b.pdf

    ##TODO update the math
    .. math::

        \begin{align*}
            g_k &= \nabla f_{i_k}(x_k) - \nabla f_{i_k} (x_{k-1}) + g_{k-1} \\
            x_{k+1} &= x_k - \eta g_k
        \end{align*}

    It is used to solve

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
                Step size for the gradient step of SARAH
                The default :code:`step_size` is :math:`\frac{1}{L}`.
    kwargs: Keyword arguments
        Arguments from the base class :class:`.Algorithm`.

    See also
    --------
    :class:`.ISTA`
    :class:`.GD`

    """

    @property
    def step_size(self):        
       return self._step_size    

    # Set default step size
    def set_step_size(self, step_size):
        
        """ Set default step size.        
        """        
        if step_size is None:
            if isinstance(self.f.L, Number):
                self.initial_step_size = 1.99/self.f.L
                self._step_size = ConstantStepSize(self.initial_step_size)
            else:
                raise ValueError("Function f is not differentiable")                        
        else:
            if isinstance(step_size, Number):
                self.initial_step_size = step_size
                self._step_size = ConstantStepSize(self.initial_step_size)
            else:
                self._step_size = step_size  

    def __init__(self, initial, f, g, step_size = None, update_frequency = None, **kwargs):

        if not isinstance(f, ApproximateGradientSumFunction):
            raise ValueError("An ApproximateGradientSumFunction is required for f, {} is passed".format(f.__class__.__name__))             
        super(SARAH, self).__init__(**kwargs)
        
        # step size
        self._step_size = None

       # initial step size for adaptive step size
        self.initial_step_size = None
        
        self.set_up(initial=initial, f=f, g=g, step_size=step_size, update_frequency=update_frequency,**kwargs)

    def set_up(self, initial, f, g, step_size, update_frequency, **kwargs): # update frequency
        """ Set up the algorithm
        """

        logging.info("{} setting up".format(self.__class__.__name__, ))
        
        self.initial = initial
        self.f = f # at the moment this is required to be of SubsetSumFunctionClass (so that data_passes member exists)
        self.g = g

        # set problem parameters    
        self.update_frequency = update_frequency
        if self.update_frequency is None:
            self.update_frequency = self.f.num_functions
        
        self.set_step_size(step_size=step_size)

        # Initialise iterates, the gradient estimator, and the temporary variables
        self.x_old = initial.copy()
        self.x = initial.copy()

        self.gradient_estimator = self.x * 0.0
        self.stoch_grad_at_iterate = self.x * 0.0
        self.stochastic_grad_difference = self.x * 0.0

        self.configured = True
        logging.info("{} configured".format(self.__class__.__name__, ))

    def update(self):

        r"""Performs a single iteration of SARAH

        .. math::
            # TODO: change maths
            \begin{cases}

            \end{cases}

        """
        
        self.approximate_gradient(self.x, out=self.gradient_estimator) 
        self.x_old = self.x.copy()
        step_size =  self.step_size(self)
        self.x.sapyb(1., self.gradient_estimator, -step_size, out = self.x)
        self.x = self.g.proximal(self.x, step_size)       
        
    def approximate_gradient(self, x, out = None):            

        update_flag = (self.iteration % (self.update_frequency) == 0)

        if update_flag is True:
            
            # update the full gradient estimator
            self.f.full_gradient(x, out=self.gradient_estimator)

            if self.iteration == 0:
                if len(self.f.data_passes) == 0:
                    self.f.data_passes.append(1)
                else:
                    self.f.data_passes[0] = 1.
            else:
                self.f.data_passes.append(self.f.data_passes[-1]+1.)
                
            if out is None:
                return self.gradient_estimator
            else:
                out = self.gradient_estimator
        else:
            
            self.f.next_function()
            self.f.functions[self.f.function_num].gradient(x, out=self.stoch_grad_at_iterate)
            self.stoch_grad_at_iterate.sapyb(1., self.f.functions[self.f.function_num].gradient(self.x_old), -1., out=self.stochastic_grad_difference)
            
            # update the data passes
            self.f.data_passes.append(round(self.f.data_passes[-1] + 1./self.f.num_functions,2))

            # Compute the output: gradient difference +  v_t
            if out is None:
                return self.stochastic_grad_difference.sapyb(self.f.num_functions, self.gradient_estimator, 1.)
            else:
                return self.stochastic_grad_difference.sapyb(self.f.num_functions, self.gradient_estimator, 1., out=out)

    
    def update_objective(self):
        """ Updates the objective
        .. math:: f(x) + g(x)
        """
        self.loss.append( self.f(self.get_output()) + self.g(self.get_output()) )            
       