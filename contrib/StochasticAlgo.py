from cil.optimisation.algorithms import Algorithm
from cil.optimisation.functions import IndicatorBox, Function
import numpy as np

##################################################
########### Design: PLAN A #######################
##################################################

class SubsetGradientAlgorithm(Algorithm):
    
    """
        Generic SubsetGradientAlgorithm Class: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9226504
        SAG, SAGA, SVRG
        
        # Maybe create a class for the Gradient memory, to decide
    """    

    # precondition = True/False and how do we format it. Not implemented atm
    
    def __init__(self, x0, function, step_size=1, **kwargs):
        
        # inherits from Algorithm base                
        super(SubsetGradientAlgorithm, self).__init__(**kwargs)        
        
        # init calls set_up
        self.set_up(x0 = x0, function = function, step_size=step_size, **kwargs)        
        
    def set_up(self, x0, function, step_size, **kwargs):
        
        # Need to check that this is a BlockFunction???
        # assume f has gradient method implemented otherwise raise Error.
        self.function = function
            
        # Check that stepsize is positive
        self.step_size = step_size
        
        # Get num of subsets
        self.num_subsets = len(function) 
        
        # initial
        self.x0 = x0
        
        # non-negative constraint, for TV will be include, to decide
        self.constraint = IndicatorBox(lower=0.0)
        
        # call memory_init method
        self.memory = self._memory_init()
        
            
    def _memory_init(self):
        """
            init of v_i's and g_bar                    
        """
        raise NotImplemented
    
    def _memory_update(self):
        """
            update of v_i's and g_bar            
        """
        raise NotImplemented
    
        
    def _approx_gradient(self, i):
        """
         Defined by the specific algorithm: \tilde{\nabla} 
         
         i: subset number
         
        """
        raise NotImplemented
                    
    def update(self):
                
        #choose random subset, int or not?
        subset_num = int(np.random.choice(self.num_subsets))
                
        sub_grad_new = self.function[i].gradient(self.x0)        
                        
        # gradient step, missing preconditioning 
        self.x0 = self.constraint(self.x0 + self.step_size * self._approx_gradient(subset_num, sub_grad_new))
        
        # memory step
        self._memory_update(subset_num, sub_grad_new)
        
    def update_objective(self):
        
        """
            objective value
            
            # Call method of BlockFunction, needs a blockdatacontainer, instead we iterate on each
            # function.
        """
                
        s = 0
        for i in range(self.num_subsets):
            s += self.function[i](self.x0)
        return s
                 

class SAGA(SubsetGradientAlgorithm):
    
    
    """
        SAGA algorithm
    """
    
    def __init__(self, **kwargs):
        
        # inherits from StochasticAlgorithm base class and calls parent method: set_up
        super(SAGA, self).__init__(**kwargs) 
        
        # default gradient init to be zero
        self.init_grad_zero = kwargs.get('init_grad_zero', True)
                     
    def _memory_init(self):
        
        """
        
            initialize subset gradient and full gradient and store in memory
        
        """
        
        self.subset_gradients = []
        self.full_gradient = self.x0.copy()*0.0
        
        if self.init_grad_zero:
            
            for i in range(self.num_subsets):
                self.subset_gradients.append(self.x0.copy()*0.0)
              
        else:
            
            for i in range(self.num_subsets):
                
                sub_grad = self.function[i].gradient(self.x0)
                self.subset_gradients.append(sub_grad)
                self.full_gradient += sub_grad
                
            # average over self.num_subsets    
            self.full_gradient /= self.num_subsets
                                                    
    
    def _memory_update(self, subset_num, sub_grad_new):
        
        """ 
        
          subset_num: the number of the subset
          
          update sub_grad of subset_num          
          
          full_gradient = full_gradient - 1/num_subsets * sub_grad_old + 1/num_subsets * sub_grad_new
          
          
        """
        
        sub_grad_old = self.subset_gradients[subset_num]
        
        self.full_gradient += (sub_grad_new - sub_grad_old)/self.num_subsets
        self.subset_gradients[subset_num] = sub_grad_new
                
    
    def _approx_gradient(self, subset_num, sub_grad_new):
        
        
        return sub_grad_new - self.subset_gradients[subset_num] + self.full_gradient
        

##################################################
########### Design: PLAN B #######################
##################################################

# Create  a GeneralisedFunction class
    
#     - has a __call__ method computing the sum etc
#     - has a "approx gradient" method, not Implemented by default
#     - has method that computes full gradient

# Create a Child classs from GeneralisedFunction class, e.g., 
#  FunctionSAGA, decide on the name
#  FunctionSAG, decide on the name
#  FunctionSVRG, decide on the name

class GeneralisedFunction(Function):
    
    def __init__(self, function, algorithm):
        pass
    
    def gradient(self):
        """
        
            Evaluate approximate grad
        
        """
        pass
        
    