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


    - (optional) 
        subset_select_function: function which takes two integers and outputs an integer
        defines how the subset is chosen at each iteration
        default is uniformly at random    

    '''
    
    def __init__(self, functions, subset_select_function=(lambda a,b: int(np.random.choice(b))), subset_init=-1, **kwargs):
        self.subset_select_function = subset_select_function
        self.subset_num = subset_init
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

        self.subset_num = self.subset_select_function(self.subset_num, self.num_subsets)


class SAGAGradientFunction(SubsetSumFunction):
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method doesn't the mathematical gradient of the sum of functions, 
    but a variance-reduced approximated gradient corresponding to the SAGA algorithm.
    More details below, in the gradient method.

    Parameters:
    -----------

    - functions: a list of functions

    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
           
    '''
        self.gradients_allocated = False
        
        
        super(SAGAGradientFunction, self).__init__(functions)

    def gradient(self, x, out=None):
        """
        Returns a variance-reduced approximate gradient, defined below.

        For f = \sum f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - compute
                subset_grad - subset_grad_old + 1/num_subsets * full_grad
                with 
                - subset_grad is the gradient of function number j at current point
                - subset_grad_old is the gradient of function number j in memory
                - full_grad is the approximation of the gradient of f in memory
            - update subset_grad and full)grad
        
        Combined with the gradient step, the algorithm is guaranteed 
        to converge if the functions f_i are convex and the step-size 
        gamma satisfies to
            gamma <= 1/(3 * max L_i)
        where the gradient of each f_i is L_i - Lipschitz.

        Reference:
        Defazio, Aaron, Bach, Francis, and Simon Lacoste-Julien. 
        "SAGA: A fast incremental gradient method with support 
        for non-strongly convex composite objectives." 
        Advances in neural information processing systems. 2014.

        NB: 
        contrarily to the convention in the above article, 
        the objective function used here is 
            \sum f_i
        and not
            1/num_subsets \sum f_i
        """

        if not self.gradients_allocated:
            self.memory_init(x) 

        # random choice of subset
        self.next_subset()

        full_grad_old = self.full_gradient

        # Compute new gradient for current subset, store in tmp2
        # subset_grad = self.function[self.subset_num].gradient(x)
        self.functions[self.subset_num].gradient(x, out=self.tmp2)

        # Compute difference between new and old gradient for current subset, store in tmp1
        # subset_grad_old = self.subset_gradients[self.subset_num]
        # subset_grad - subset_grad_old
        self.tmp2.axpby(1., -1., self.subset_gradients[self.subset_num], out=self.tmp1)

        # Compute output subset_grad - subset_grad_old + 1/num_subsets * full_grad
        if out is None:
            ret = 0.0 * self.tmp1
            self.tmp1.axpby(1., 1./self.num_subsets, full_grad_old, out=ret)
        else:
            self.tmp1.axpby(1., 1./self.num_subsets, full_grad_old, out=out)

        if out is None:
            return ret

        # update subset gradient: store subset_grad in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.subset_num].fill(self.tmp2)

        # update full gradient: needs subset_grad - subset_grad_old, which is stored in tmp1
        # full_gradient = full_gradient + subset_grad - subset_grad_old
        self.full_gradient.axpby(1., 1., self.tmp1, out=self.full_gradient)

        if out is None:
            return ret


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