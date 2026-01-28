from cil.optimisation.algorithms import Algorithm
import numpy as np
import logging


class ProxSkip(Algorithm):
    

    r"""Proximal Skip  (ProxSkip) algorithm, see "ProxSkip: Yes! Local Gradient Steps Provably Lead to Communication Acceleration! Finally!†"
    
        Parameters
        ----------

        initial : DataContainer
                  Initial point for the ProxSkip algorithm. 
        f : Function
            A smooth function with Lipschitz continuous gradient.
        g : Function
            A convex function with a "simple" proximal.
        prob : positive :obj:`float`
             Probability to skip the proximal step. If :code:`prob=1`, proximal step is used in every iteration.             
        step_size : positive :obj:`float`
            Step size for the ProxSkip algorithm and is equal to 1./L where L is the  Lipschitz constant for the gradient of f.          
            
     """


    def __init__(self, initial, f, g, step_size, prob, seed=None, **kwargs):
        """ Set up of the algorithm
        """        

        super(ProxSkip, self).__init__(**kwargs)

        self.f = f # smooth function
        self.g = g # proximable
        self.step_size = step_size
        self.prob = prob
        self.rng = np.random.default_rng(seed)
        self.set_up(initial, f, g, step_size, prob, **kwargs)
        self.thetas = []
 
                  
    def set_up(self, initial, f, g, step_size, prob, **kwargs):
        
        logging.info("{} setting up".format(self.__class__.__name__, ))        
        
        self.initial = initial[0]

        self.x = initial.copy()   
        self.xhat_new = initial.copy()
        self.x_new = initial.copy()
        self.ht = initial.copy() 

        self.configured = True
        
        logging.info("{} configured".format(self.__class__.__name__, ))
                                  

    def update(self):
        r""" Performs a single iteration of the ProxSkip algorithm        
        """
        
        self.f.gradient(self.x, out=self.xhat_new)
        self.xhat_new -= self.ht
        self.x.sapyb(1., self.xhat_new, -self.step_size, out=self.xhat_new)

        theta = self.rng.random() < self.prob 
        # convention: use proximal in the first iteration
        if self.iteration==0:
            theta = 1
        self.thetas.append(theta)

        if theta==1:
            # Proximal step is used
            self.g.proximal(self.xhat_new - (self.step_size/self.prob)*self.ht, self.step_size/self.prob, out=self.x_new)
            self.ht.sapyb(1., (self.x_new - self.xhat_new), (self.prob/self.step_size), out=self.ht)            
        else:
            self.x_new.fill(self.xhat_new)

    def _update_previous_solution(self):
        """ Swaps the references to current and previous solution based on the :func:`~Algorithm.update_previous_solution` of the base class :class:`Algorithm`.
        """
        tmp = self.x_new
        self.x = self.x_new
        self.x = tmp

    def get_output(self):
        " Returns the current solution. "
        return self.x        
      
                  
    def update_objective(self):

        """ Updates the objective

        .. math:: f(x) + g(x)

        """        

        fun_g = self.g(self.x)
        fun_f = self.f(self.x)
        p1 = fun_f + fun_g
        self.loss.append( p1 )

