from cil.optimisation.algorithms import Algorithm
import numpy as np
import logging
from warnings import warn


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
            Step size for the ProxSkip algorithm. It is equal to 1./L for strongly convex f and 2./L for convex f, where L is the  Lipschitz constant for the gradient of f.          
            
     """


    def __init__(self, initial, f, g, step_size, prob, seed=None, **kwargs):
        """ Set up of the algorithm
        """        

        super(ProxSkip, self).__init__(**kwargs)

        self.f = f # smooth function
        self.g = g # proximable
        self.step_size = step_size
        self.prob = prob
        self.rng = np.random.default_rng(seed=seed)
        self.thetas = []

        if self.prob<=0:
            raise ValueError("Need a positive probability")
        if self.prob==1:
            raise warn("If p=1, ProxSkip is equivalent to ISTA/PGD. Please use ISTA/PGD to avoid computing updates of the control variate that is not used.")
        
        self.set_up(initial, f, g, step_size, prob, **kwargs)
        
                  
    def set_up(self, initial, f, g, step_size, prob, **kwargs):
        
        logging.info("{} setting up".format(self.__class__.__name__, ))        
        
        ## TODO better to use different initials for x and h.        
        self.initial = initial
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
            theta = True
        self.thetas.append(theta)

        if theta:
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
        

