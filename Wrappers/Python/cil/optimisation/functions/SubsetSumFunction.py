from cil.optimisation.functions import Function
import numpy as np

class AveragedSumFunction(Function):
    
    """ AveragedSumFunction represents the sum of :math:`n\geq2` functions
    
    .. math:: (1/n*(F_{1} + F_{2} + ... + F_{n}))(x)  = 1/n*( F_{1}(x) + F_{2}(x) + ... + F_{n}(x))

    		    
    Parameters		
    ----------		
    *functions : Functions		
                 Functions to set up a :class:`.SumFunction`		
    Raises		
    ------		
    ValueError		
            If the number of function is strictly less than 2.		    
    """

    
    
    def __init__(self, *functions ):
                
        super(AveragedSumFunction, self).__init__()        
        if len(functions) < 2:
            raise ValueError('At least 2 functions need to be passed')
        self.functions = functions
        self.num_functions = len(self.functions)

    @property
    def L(self):
        """Returns the Lipschitz constant for the gradient of the  AveragedSumFunction		       
        		
        .. math:: L = \frac{1}{n} \sum_{i=1}^n L_{i}		
        where :math:`L_{i}` is the Lipschitz constant of the gradient of the smooth function :math:`F_{i}`.		
        		
        """
        
        L = 0.
        for f in self.functions:
            if f.L is not None:
                L += f.L
            else:
                L = None
                break
        self._L = L
            
        return 1/self.num_functions * self._L

        
    @L.setter
    def L(self, value):
        # call base class setter
        super(AveragedSumFunction, self.__class__).L.fset(self, value )

    @property
    def Lmax(self):
        """Returns the maximum Lipschitz constant for the AveragedSumFunction		
        		
        .. math:: L = \max_{i}\{L_{i}\}		
        where :math:`L_{i}` is the Lipschitz constant of the gradient of the smooth function :math:`F_{i}`.		
        		        
        """
        
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
        super(AveragedSumFunction, self.__class__).Lmax.fset(self, value )

    def __call__(self,x):
        r"""Returns the value of the averaged sum of functions at :math:`x`.		
        		
        .. math:: ( \frac{1}{n}(F_{1} + F_{2} + ... + F_{n}))(x) = \frac{1}{n} *( F_{1}(x) + F_{2}(x) + ... + F_{n}(x))
                		
        """ 
        ret = 0.
        for f in self.functions:
            ret += f(x)
        return 1/self.num_functions * ret


    def gradient(self, x, out=None):
        
        r"""Returns the value of the averaged sum of the gradient of functions at :math:`x`, if all of them are differentiable.
        
        .. math::(1/n* (F'_{1} + F'_{2} + ... + F'_{n}))(x) = 1/n * (F'_{1}(x) + F'_{2}(x) + ... + F'_{n}(x))
        
        """
        
        if out is None:            
            for i,f in enumerate(self.functions):
                if i == 0:
                    ret = 1/self.num_functions * f.gradient(x)
                else:
                    ret += 1/self.num_functions * f.gradient(x)
            return ret
        else:
            for i,f in enumerate(self.functions):
                if i == 0:
                    f.gradient(x, out=out)
                    out *= 1/self.num_functions
                else:
                    out +=  1/self.num_functions * f.gradient(x)

 

class SubsetSumFunction(AveragedSumFunction):
    
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method is implemented in children classes and allows to return an approximation of the gradient based on subset gradients.

    Parameters:
    -----------

    - functions: f1, f2, 

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
        return self.num_functions
        
    def _full_gradient(self, x, out=None):
        '''Returns  (averaged) full gradient'''
        return super(SubsetSumFunction, self).gradient(x, out=out)
        
    def gradient(self, x, out=None):
        """        
            maybe calls _approx_grad inside        
        """
        raise NotImplemented


    def subset_gradient(self, x, subset_num,  out=None):
        '''Returns (non-averaged) partial gradient'''

        return self.functions[subset_num].gradient(x)


    def next_subset(self):

        self.subset_num = self.subset_select_function(self.subset_num, self.num_subsets)


class SAGAFunction(SubsetSumFunction):
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method doesn't the mathematical gradient of the sum of functions, 
    but a variance-reduced approximated gradient corresponding to the SAGA algorithm.
    More details below, in the gradient method.

    Parameters:
    -----------

    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    - (optional)
        gradient_initialization_point: point to initialize the gradient of each subset
        and the full gradient
        default None
           
    '''

    def __init__(self, functions, precond=None, gradient_initialization_point=None, **kwargs):
        
        super(SAGAFunction, self).__init__(functions)

        if gradient_initialization_point is None:
            self.gradients_allocated = False
        else:
            self.subset_gradients = [ fi.gradient(gradient_initialization_point) for i, fi in enumerate(functions)]
            self.full_gradient = 1/self.num_subsets * sum(self.subset_gradients)
            self.tmp1 = gradient_initialization_point * 0.0
            self.tmp2 = gradient_initialization_point * 0.0
            self.gradients_allocated = True
        self.precond = precond

    def gradient(self, x, out=None):
        """
        Returns a variance-reduced approximate gradient, defined below.

        For f = \sum f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - compute
                subset_grad - subset_grad_old +  full_grad
                with 
                - subset_grad is the gradient of function number j at current point
                - subset_grad_old is the gradient of function number j in memory
                - full_grad is the approximation of the gradient of f in memory
            - update subset_grad and full_grad
        
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
        # tmp1 = subset_grad - subset_grad_old
        self.tmp2.axpby(1., -1., self.subset_gradients[self.subset_num], out=self.tmp1)

        # Compute output subset_grad - subset_grad_old +  full_grad
        if out is None:
            ret = 0.0 * self.tmp1
            self.tmp1.axpby(1., 1., full_grad_old, out=ret)
        else:
            self.tmp1.axpby(1., 1., full_grad_old, out=out)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        # update subset gradient: store subset_grad in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.subset_num].fill(self.tmp2)

        # update full gradient by adding to 1/num_subsets *  (subset_grad - subset_grad_old) to last value
        # full_gradient = full_gradient + 1/num_subsets * (subset_grad - subset_grad_old)
        # ubset_grad - subset_grad_old is stored in tmp1
        self.full_gradient.axpby(1., 1./self.num_subsets, self.tmp1, out=self.full_gradient)

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
    
    def memory_reset(self):
        """        
            resets subset gradients and full gradient in memory.

        """
        if self.gradients_allocated == True:
            del(self.subset_gradients)
            del(self.full_gradient)
            del(self.tmp1)
            del(self.tmp2)

            self.gradients_allocated = False
            
            
class SGDFunction(SubsetSumFunction):
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.
    The `gradient` method doesn't the mathematical gradient of the sum of functions,
    but a approximated gradient corresponding to the minibatch SGD algorithm. --Billy 23/2/2022
    More details below, in the gradient method.
    Parameters:
    -----------
    - (optional)
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    '''

    def __init__(self, functions, precond=None, **kwargs):

        self.gradients_allocated = False
        self.precond = precond

        super(SGDFunction, self).__init__(functions)

    def gradient(self, x, out=None):
        """
        Returns a vanilla stochastic gradient estimate, defined below.
        For f = \sum f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - compute
                subset_grad
                with
                - subset_grad is the gradient of function number j at current point

        where the gradient of each f_i is L_i - Lipschitz.
        """

        output = x * 0.0
        self.next_subset()
        self.functions[self.subset_num].gradient(x, out=output)

        return output
