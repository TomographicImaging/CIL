from cil.optimisation.functions import SumFunction
from cil.optimisation.utilities import RandomSampling
import numbers
class ApproximateGradientSumFunction(SumFunction):

    r"""ApproximateGradientSumFunction represents the following sum 
    
    .. math:: \sum_{i=1}^{n} F_{i} = (F_{1} + F_{2} + ... + F_{n})

    where :math:`n` is the number of functions.

    Parameters:
    -----------
    functions : list(functions) 
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method.
    selection : :obj:`method`, Default = :code:`RandomSampling.uniform(len(functions))`. 
               Determines how the next function is selected using :class:`RandomSampling`.               
                
    Note
    ----
        
    The :meth:`~ApproximateGradientSumFunction.gradient` computes the `gradient` of only one function of a batch of functions 
    depending on the :code:`selection` method. The selected function(s) is  the :meth:`~SubsetSumFunction.next_subset` method.
    
    Example
    -------

    .. math:: \sum_{i=1}^{n} F_{i}(x) = \sum_{i=1}^{n}\|A_{i} x - b_{i}\|^{2}

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets))   
    >>> f = ApproximateGradientSumFunction(list_of_functions) 

    >>> list_of_functions = [LeastSquares(Ai, b=bi)] for Ai,bi in zip(A_subsets, b_subsets)) 
    >>> selection = RandomSampling.random_shuffle(len(list_of_functions))
    >>> f = ApproximateGradientSumFunction(list_of_functions, selection=selection)   
  

    """
    
    def __init__(self, functions, selection=None):    
                        
        self.functions_used = []   
        if selection is None:
            self.selection = RandomSampling.uniform(len(functions))
        else:
            self.selection = selection
            
        super(ApproximateGradientSumFunction, self).__init__(*functions)            
  
    def __call__(self, x):

        r"""Returns the value of the sum of functions at :math:`x`.		

        .. math:: \sum_{i=1}^{n}(F_{i}(x)) = (F_{1}(x) + F_{2}(x) + ... + F_{n}(x))

        """
                		
        return super(ApproximateGradientSumFunction, self).__call__(x)      
        
    def full_gradient(self, x, out=None):

        r""" Computes the full gradient at :code:`x`. It is the sum of all the gradients for each function. """
        return super(ApproximateGradientSumFunction, self).gradient(x, out=out)
        
    def approximate_gradient(self, function_num, x,  out=None):

        """ Computes the approximate gradient for each selected function at :code:`x`."""      
        raise NotImplemented

    def gradient(self, x, out=None):

        """ Computes the gradient for each selected function at :code:`x`."""   
        self.next_function()

        # single function 
        if isinstance(self.function_num, numbers.Number):
            return self.approximate_gradient(self.function_num, x, out=out)
        else:            
            raise ValueError("Batch gradient is not implemented")
               
    def next_function(self):
        
        """ Selects the next function or the next batch of functions from the list of :code:`functions` using the :code:`selection`."""        
        self.function_num = next(self.selection)
        
        # append each function used at this iteration
        self.functions_used.append(self.function_num)