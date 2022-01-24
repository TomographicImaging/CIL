from cil.optimisation.functions import Function
import numpy as np

# Temporarily redefine SumFunction (PR #1093)
class SumFunction(Function):
    
    """ SumFunction represents the sum of two functions
    
    .. math:: (F_{1} + F_{2} + ... + F_{n})(x)  = F_{1}(x) + F_{2}(x) + ... + F_{n}(x)
    
    """
    
    def __init__(self, *functions ):
                
        super(SumFunction, self).__init__()        
        if len(functions) < 2:
            raise ValueError('At least 2 functions need to be passed')
        self.functions = functions

    @property
    def L(self):
        '''Lipschitz constant of the gradient of the sum of the functions'''
        
        L = 0.
        for f in self.functions:
            if f.L is not None:
                L += f.L
            else:
                L = None
                break
        self._L = L
            
        return self._L

        
    @L.setter
    def L(self, value):
        # call base class setter
        super(SumFunction, self.__class__).L.fset(self, value )

    def __call__(self,x):
        r"""Returns the value of the sum of functions :math:`F_{1}`,  :math:`F_{2}` ... :math:`F_{n}`at x
        
        .. math:: (F_{1} + F_{2} + ... + F_{n})(x) = F_{1}(x) + F_{2}(x) + ... + F_{n}(x)
                
        """  
        ret = 0.
        for f in self.functions:
            ret += f(x)
        return ret

    @property
    def Lmax(self):
        '''Maximum of the Lipschitz constants of the gradients of each function in the sum'''
        
        l = []
        for f in self.functions:
            if f.L is not None:
                l.append(f.L)
            else:
                l = None
                break
        self._Lmax = max(l)
            
        return self._Lmax

        
    @Lmax.setter
    def Lmax(self, value):
        # call base class setter
        super(SumFunction, self.__class__).Lmax.fset(self, value )

    def __call__(self,x):
        r"""Returns the value of the sum of functions :math:`F_{1}`,  :math:`F_{2}` ... :math:`F_{n}`at x
        
        .. math:: (F_{1} + F_{2} + ... + F_{n})(x) = F_{1}(x) + F_{2}(x) + ... + F_{n}(x)
                
        """  
        ret = 0.
        for f in self.functions:
            ret += f(x)
        return ret


    def gradient(self, x, out=None):
        
        r"""Returns the value of the sum of the gradient of functions :math:`F_{1}`,  :math:`F_{2}` ... :math:`F_{n}` at x, 
        if all of them are differentiable
        
        .. math:: (F'_{1} + F'_{2} + ... + F'_{n})(x) = F'_{1}(x) + F'_{2}(x) + ... + F'_{n}(x)
        
        """
        
        if out is None:            
            for i,f in enumerate(self.functions):
                if i == 0:
                    ret = f.gradient(x)
                else:
                    ret += f.gradient(x)
            return ret
        else:
            for i,f in enumerate(self.functions):
                if i == 0:
                    f.gradient(x, out=out)
                else:
                    out += f.gradient(x)
    def __add__(self, other):
        
        """ Returns the sum of the functions.
        
            Cases: a) the sum of two functions :math:`(F_{1}+F_{2})(x) = F_{1}(x) + F_{2}(x)`
                   b) the sum of a function with a scalar :math:`(F_{1}+scalar)(x) = F_{1}(x) + scalar`
        """
        
        if isinstance(other, SumFunction):
            functions = list(self.functions) + list(other.functions)
            return SumFunction(*functions)
        elif isinstance(other, Function):
            functions = list(self.functions)
            functions.append(other)
            return SumFunction(*functions)
        else:
            return super(SumFunction, self).__add__(other)


class SubsetSumFunction(SumFunction):
    
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method is implemented in children classes and allows to return an approximation of the gradient based on subset gradients.

    Parameters:
    -----------



    '''
    
    def __init__(self, functions, **kwargs):
        # should not have docstring
        super(SubsetSumFunction, self).__init__(*functions)
        
    @property
    def num_subsets(self):
        return len(self.functions)
        
    def _full_gradient(self, x, out=None):
        '''Return full gradient'''
        return super(SubsetSumFunction, self).gradient(x, out=out)
        
    def gradient(self, x, out=None):
        """        
            maybe calls _approx_grad inside        
        """
        raise NotImplemented


    def subset_gradient(self, x, subset_num,  out=None):

        return self.functions[subset_num].gradient(x)


    def next_subset(self):

        raise NotImplemented


class SAGAGradientFunction(SubsetSumFunction):

    def __init__(self, functions):

        self.gradients_allocated = False
        
        
        super(SAGAGradientFunction, self).__init__(functions)

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