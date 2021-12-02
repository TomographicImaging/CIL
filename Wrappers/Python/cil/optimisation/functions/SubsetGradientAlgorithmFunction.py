from cil.optimisation.functions import Function, SumFunction
import numpy as np

class SubsetGradientAlgorithmFunction(SumFunction):
    
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method is implemented in children classes and allows to return an approximation of the gradient based on subset gradients.

    Parameters:
    -----------



    '''
    
    # name suggestions: CustomGradientFunction, 
    #                   A name with the subsets word
    #     

    
    def __init__(self, functions, **kwargs):
        # should not have docstring
        super(SubsetGradientAlgorithmFunction, self).__init__(*functions)
        
    @property
    def num_subsets(self):
        return len(self.functions)
        
    def _full_gradient(self, x, out=None):
        '''Return full gradient'''
        return super(SubsetGradientAlgorithmFunction, self).gradient(x, out=out)
        
    def gradient(self, x, out=None):
        """        
            maybe calls _approx_grad inside        
        """
        raise NotImplemented


    def subset_gradient(self, x, subset_num,  out=None):

        return self.functions[subset_num].gradient(x)


    def next_subset(self):

        raise NotImplemented


class SAGA_Function(SubsetGradientAlgorithmFunction):

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
        self.functions[self.subset_num].gradient(x, out=self.tmp2)

        # the following line computes these and stores the result in tmp1
        # subset_grad = self.num_subsets * self.function[self.subset_num].gradient(x)
        # subset_grad - subset_grad_old
        self.tmp2.axpby(self.num_subsets, -1., self.subset_gradients[self.subset_num], out=self.tmp1)
        # store the new subset_grad in self.subset_gradients[self.subset_num]
        self.tmp2.multiply(self.num_subsets, out=subset_grad_old)
        
        if out is None:
            ret = self.tmp1.add(full_grad_old)
        else:
            self.tmp1.add(full_grad_old, out=out)
        # update full gradient, which needs subset_grad - subset_grad_old, which is stored in tmp1
        self.full_gradient.axpby(1., 1/self.num_subsets, self.tmp1, out=self.full_gradient)

        if out is None:
            return ret


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
        self.tmp2 = x * 0.0

        self.gradients_allocated = True