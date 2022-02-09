from cil.optimisation.algorithms import Algorithm
import warnings

class ISTA(Algorithm):
    
    r''' Iterative Shrinkage-Thresholding Algorithm 
    
    Problem : 
    
    .. math::
    
      \min_{x} f(x) + g(x)
    
    |
    
    Parameters :
        
      :param initial: Initial guess ( Default initial = 0)
      :param f: Differentiable function
      :param g: Convex function with " simple " proximal operator


    References :

        P. L. Combettes and V. A. L. Erie, 2005.
        Signal Recovery by Proximal Forward-Backward Splitting. 
        Multiscale Model. Simul., vol. 4, no. 4, pp. 1168–1200
      
        Beck, A. and Teboulle, M., 2009. A fast iterative shrinkage-thresholding 
        algorithm for linear inverse problems. 
        SIAM journal on imaging sciences,2(1), pp.183-202.

    Convergence conditions in Combettes-Erie 2009:
        f and g convex and f differentiable with L-Lipschitz gradient
        step-size < 2/L
    '''
    
    
    def __init__(self, initial, f, g, step_size = None, **kwargs):
        
        '''ISTA algorithm creator 
        
        initialisation can be done at creation time if all 
        proper variables are passed or later with set_up
        
        Optional parameters:

        :param initial: Initial guess ( Default initial = 0)
        :param f: Differentiable function
        :param g: Convex function with " simple " proximal operator'''
        
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
        '''initialisation of the algorithm

        :param initial: Initial guess ( Default initial = 0)
        :param f: Differentiable function
        :param g: Convex function with " simple " proximal operator'''

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
        
        # gradient step
        self.f.gradient(self.x_old, out=self.x)
        self.x *= -self.step_size
        self.x += self.x_old
        
        # proximal step
        self.g.proximal(self.x, self.step_size, out=self.x)

        # update
        self.x_old.fill(self.x)
        
    def update_objective(self):
        self.loss.append( self.f(self.x) + self.g(self.x) )    
