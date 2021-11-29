from cil.optimisation.functions import Function, SumFunction
import numpy as np

class StochasticGradientAlgorithmFunction(SumFunction):
    
    '''Class for use as objective function in Stochastic Gradient type algorithms such as GD
    
    Parameters:
    -----------



    '''
    
    # name suggestions: CustomGradientFunction, 
    #                   A name with the subsets word
    #     

    
    def __init__(self, functions, **kwargs):
        # should not have docstring
        super(StochasticGradientAlgorithmFunction, self).__init__(*functions)
        
    @property
    def num_subsets(self):
        return len(self.functions)
        
    def _full_gradient(self, x, out=None):
        '''Return full gradient'''
        return super(StochasticGradientAlgorithmFunction, self).gradient(x, out=out)
        
    def gradient(self, x, out=None):
        """        
            maybe calls _approx_grad inside        
        """
        raise NotImplemented


    def subset_gradient(self, x, subset_num,  out=None):

        return self.functions[subset_num].gradient(x)


    def next_subset(self):

        raise NotImplemented


class SAGA_Function(StochasticGradientAlgorithmFunction):

    def __init__(self, functions):

        self.gradients_allocated = False
        
        
        super(SAGA_Function, self).__init__(functions)

    def gradient(self, x, out=None):

        if not self.gradients_allocated:
            self.memory_init(x) 

        # random choice of subset
        self.next_subset()

        subset_grad_old = self.subset_gradients[self.subset_num]

        full_grad_old = self.full_gradient

        # This is step 6 of the SAGA algo, and we multiply by the num_subsets to take care the (1/n) weight
        # step below to be optimised --> multiplication
        # subset_grad = self.num_subsets * self.functions[self.subset_num].gradient(x)
        
        # subset_grad gradient of the current function
        self.functions[self.subset_num].gradient(x, out=self.current_gradient)

        # the following line computes these and stores the result in tmp1
        # subset_grad = self.num_subsets * self.function[self.subset_num].gradient(x)
        # subset_grad - subset_grad_old
        self.current_gradient.axpby(self.num_subsets, -1., subset_grad_old, out=self.tmp1)
        # store the new subset_grad in self.subset_gradients[self.subset_num] 
        self.tmp1.add(full_grad_old, out=subset_grad_old)
        
        # update full gradient, which needs subset_grad - subset_grad_old, which is stored in tmp2
        self.full_gradient.axpby(1., 1/self.num_subsets, self.tmp1, out=self.full_gradient)
        
        return self.current_gradient


    # def memory_update(self, subset_grad):

    #     # step below to be optimised --> div
    #     self.full_gradient += (subset_grad - self.subset_gradients[self.subset_num])/self.num_subsets
    #     self.subset_gradients[self.subset_num] = subset_grad 
        

    def next_subset(self):
        
        self.subset_num = int(np.random.choice(self.num_subsets))

    def memory_init(self, x):
        
        """        
            initialize subset gradient (v_i_s) and full gradient (g_bar) and store in memory.

        """
        
        # this is the memory init = subsets_gradients + full gradient
        self.subset_gradients = [ x * 0.0 for _ in range(self.num_subsets)]
        self.full_gradient = x * 0.0
        self.tmp1 = x * 0.0
        self.current_gradient = x * 0.0

        self.gradients_allocated = True
