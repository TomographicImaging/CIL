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

    - functions: F_{1}, F_{2}, ..., F_{num_subsets}

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

    The `gradient` method does not correspond to the mathematical gradient of a sum of functions, 
    but rather to a variance-reduced approximated gradient corresponding to the minibatch SAGA algorithm.
    More details can be found below, in the gradient method.

    Parameters:
    -----------

    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    - (optional)
        gradient_initialisation_point: point to initialize the gradient of each subset
        and the full gradient
        default None
           
    '''

    def __init__(self, functions, precond=None, gradient_initialisation_point=None, **kwargs):
        
        super(SAGAFunction, self).__init__(functions)

        self.gradient_initialisation_point = gradient_initialisation_point
        self.gradients_allocated = False
        self.precond = precond

    def gradient(self, x, out=None):
        """
        Returns a variance-reduced approximate gradient, defined below.

        For f = 1/num_subsets \sum_{i=1}^num_subsets F_{i}, the output is computed as follows:
            - choose a subset j with the method next_subset()
            - compute
                subset_gradient - subset_gradient_in_memory +  full_gradient
                where
                - subset_gradient is the gradient of the j-th function at current iterate
                - subset_gradient_in_memory is the gradient of the j-th function, in memory
                - full_gradient is the approximation of the gradient of f in memory,
                    computed as full_gradient = 1/num_subsets \sum_{i=1}^num_subsets subset_gradient_in_memory_{i}
            - update subset_gradient and full_gradient
            - this gives an unbiased estimator of the gradient
        
        Combined with the gradient step, the algorithm is guaranteed to converge if 
        the functions f_i are convex and the step-size gamma satisfies
            gamma <= 1/(3 * max L_i)
        where L_i is the Lipschitz constant of the gradient of F_{i}

        Reference:
        Defazio, Aaron; Bach, Francis; Lacoste-Julien, Simon 
        "SAGA: A fast incremental gradient method with support 
        for non-strongly convex composite objectives." 
        Advances in neural information processing systems. 2014.
        """

        if not self.gradients_allocated:
            self.memory_init(x) 

        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()

        # Compute gradient for current subset and current iterate. store in tmp1
        # tmp1 = gradient F_{subset_num} (x)
        self.functions[self.subset_num].gradient(x, out=self.tmp1)

        # Compute the difference between the gradient of function subset_num at current iterate and the subset gradient in memory. store in tmp2
        # tmp2 = gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}
        self.tmp1.axpby(1., -1., self.subset_gradients[self.subset_num], out=self.tmp2)

        # Compute the output : tmp2 + full_gradient
        if out is None:
            ret = 0.0 * self.tmp2
            self.tmp2.axpby(1., 1., self.full_gradient, out=ret)
        else:
            self.tmp2.axpby(1., 1., self.full_gradient, out=out)

        # Apply preconditioning to the computed approximate gradient direction
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.subset_num].fill(self.tmp1)

        # Update the full gradient estimator: add 1/num_subsets * (gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}) to the current full_gradient
        self.full_gradient.axpby(1., 1./self.num_subsets, self.tmp2, out=self.full_gradient)

        if out is None:
            return ret


    def memory_init(self, x):
        """        
            initialize subset gradient (v_i_s) and full gradient (g_bar) and store in memory.

        """
        
        # If the initialisation point is not provided, set it to 0
        if self.gradient_initialisation_point is None:
            self.subset_gradients = [ x * 0.0 for _ in range(self.num_subsets)]
            self.full_gradient = x * 0.0
        # Otherwise, initialise subset gradients in memory and the full gradient at the provided gradient_initialisation_point
        else:
            self.subset_gradients = [ fi.gradient(self.gradient_initialisation_point) for i, fi in enumerate(self.functions)]
            self.full_gradient = 1/self.num_subsets * sum(self.subset_gradients)

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

    The `gradient` method does not correspond to the mathematical gradient of a sum of functions, 
    but rather to a variance-reduced approximated gradient corresponding to the minibatch SGD algorithm.
    More details can be found below, in the gradient method.

    Parameters:
    -----------

    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    - (optional)
        gradient_initialisation_point: point to initialize the gradient of each subset
        and the full gradient
        default None
           
    '''
  
    def __init__(self, functions, precond=None, **kwargs):

        super(SGDFunction, self).__init__(functions)
        self.precond = precond

    def gradient(self, x, out=None):
        """
        Returns a vanilla stochastic gradient estimate, defined below.
        For f = 1/num_subsets \sum_{i=1}^num_subsets f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - compute the gradient of the j-th function at current iterate
            - this gives an unbiased estimator of the gradient

        """

        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()

        # Compute new gradient for current subset, store in ret
        if out is None:
            ret = 0.0 * x
            self.functions[self.subset_num].gradient(x, out=ret)
        else:
            self.functions[self.subset_num].gradient(x, out=out)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        if out is None:
            return ret

class SAGFunction(SAGAFunction):
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method does not correspond to the mathematical gradient of a sum of functions, 
    but rather to a variance-reduced approximated gradient corresponding to the minibatch SAG algorithm.
    More details can be found below, in the gradient method.

    Parameters:
    -----------

    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    - (optional)
        gradient_initialisation_point: point to initialize the gradient of each subset
        and the full gradient
        default None
           
    '''
    def __init__(self, functions, precond=None, gradient_initialisation_point=None, **kwargs):
        
        super(SAGFunction, self).__init__(functions)

    def gradient(self, x, out=None):
        """
        Returns a variance-reduced approximate gradient, defined below.

        For f = 1/num_subsets \sum_{i=1}^num_subsets F_{i}, the output is computed as follows:
            - choose a subset j with the method next_subset()
            - compute
                1/num_subsets(subset_gradient - subset_gradient_old) +  full_gradient
                where
                - subset_gradient is the gradient of the j-th function at current iterate
                - subset_gradient_in_memory is the gradient of the j-th function, in memory
                - full_gradient is the approximation of the gradient of f in memory,
                    computed as full_gradient = 1/num_subsets \sum_{i=1}^num_subsets subset_gradient_in_memory_{i}
            - update subset_gradient and full_gradient
            - this gives a biased estimator of the gradient
        
        Reference:
        Schmidt, Mark; Le Roux, Nicolas; Bach, Francis. 
        "Minimizing finite sums with the stochastic average gradient."
        Mathematical Programming (162) pp.83-112. 2017.
        """

        if not self.gradients_allocated:
            self.memory_init(x) 

        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()

        # Compute gradient for current subset and current iterate. store in tmp1
        # tmp1 = gradient F_{subset_num} (x)
        self.functions[self.subset_num].gradient(x, out=self.tmp1)

        # Compute the difference between the gradient of function subset_num at current iterate and the subset gradient in memory. store in tmp2
        # tmp2 = gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}
        self.tmp1.axpby(1., -1., self.subset_gradients[self.subset_num], out=self.tmp2)

        # Compute the output : 1/num_subsets * tmp2 + full_gradient
        if out is None:
            ret = 0.0 * self.tmp2
            self.tmp2.axpby(1./self.num_subsets, 1., self.full_gradient, out=ret)
        else:
            self.tmp2.axpby(1./self.num_subsets, 1., self.full_gradient, out=out)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.subset_num].fill(self.tmp1)

        # Update the full gradient estimator: add 1/num_subsets * (gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}) to the current full_gradient
        self.full_gradient.axpby(1., 1./self.num_subsets, self.tmp2, out=self.full_gradient)

        if out is None:
            return ret

class SVRGFunction(SubsetSumFunction):
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.

    The `gradient` method does not correspond to the mathematical gradient of a sum of functions, 
    but rather to a variance-reduced approximated gradient corresponding to the minibatch SVRG algorithm.
    More details can be found below, in the gradient method.

    Parameters:
    -----------

    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
    - (optional)
        gradient_initialisation_point: point to initialize the gradient of each subset
        and the full gradient
        default None
           
    '''

    def __init__(self, functions, precond=None, update_frequency = 2, **kwargs):

        super(SVRGFunction, self).__init__(functions)

        self.gradients_allocated = False
    
        self.precond = precond
        self.update_frequency = update_frequency

        # initialise the internal iteration counter. Used to check when to update the full gradient
        self.iter = 0

    def gradient(self, x, out=None):
        """
        Returns a variance-reduced approximate gradient, defined below.
        For f = 1/num_subsets \sum_{i=1}^num_subsets f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - check if full_grad_at_snapshot and snapshot should be updated
            - if they shouldn't then compute
                subset_gradient - subset_gradient_at_snapshot + full_grad_at_snapshot
                where 
                - subset_grad is the gradient of function number j at current_iterate
                - subset_grad_at_snapshot is the gradient of function number j at snapshot in memory
                    computed as full_gradient = 1/num_subsets \sum_{i=1}^num_subsets subset_gradient_{i} (snapshot)
            - otherwise update full_grad_at_snapshot and snapshot
            - this gives an unbiased estimator of the gradient
        
        Combined with the gradient step, the algorithm is guaranteed 
        to converge if the functions f_i are convex and the step-size 
        gamma satisfies
            gamma <= 1/(4 * max L_i (P + 2))
        where the gradient of each f_i is L_i - Lipschitz, and P is the number of inner iterations, that is, update_frequency * num_sebsets

        Reference:
        Johnson, Rie; Zhang, Tong. 
        "Accelerating stochastic gradient descent using predictive variance reduction." 
        Advances in neural information processing systems. 2013.
        """

        if not self.gradients_allocated:
            self.memory_init(x)         
            
        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()

        # In the first iteration and if the snapshot image has just been updated, the current iterate and snapshot will be the same
        # thus tmp2 = 0
        if self.iter == 0 or self.iter % (self.update_frequency * self.num_subsets) == 0:
            self.tmp2 = 0.0 * self.full_gradient
        # Otherwise, compute the difference between the subset gradient at current iterate and at the snapshot in memory    
        else:
            # Compute new gradient for current subset, store in tmp1
            # tmp1 = gradient F_{subset_num} (x)
            self.functions[self.subset_num].gradient(x, out=self.tmp1) 

            # Compute difference between current subset function gradient at current iterate (tmp1) and at snapshot, store in tmp2
            # tmp2 = gradient F_{subset_num} (x) - gradient F_{subset_num} (snapshot)
            self.tmp1.axpby(1., -1., self.functions[self.subset_num].gradient(self.snapshot), out=self.tmp2) 

        # Compute the output: tmp2 + full_grad
        if out is None:
            ret = 0.0 * self.tmp2
            self.tmp2.axpby(1., 1., self.full_gradient, out=ret)
        else:
            self.tmp2.axpby(1., 1., self.full_gradient, out=out)

        self.iter += 1

        # Check whether to update the full gradient and snapshot image
        # This happens once every update_frequency epochs, i.e. if the current iterate is a multiple of update_frequency * num_subsets 
        # Setting self.update_frequency to inf indicates that we never want to conduct the update
        if np.isinf(self.update_frequency) == False and self.iter % (self.update_frequency * self.num_subsets) == 0: 
            self.memory_update(x)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        if out is None:
            return ret

    def memory_init(self, x):
        
        """        
            initialise the full gradient and the snapshot, and store in memory.
        """
        # Setting the gradient variables with the correct acquisition geometry
        self.full_gradient = x * 0.0
        self.tmp2 = x * 0.0
        self.tmp1 = x * 0.0

        # Initialise the gradient and the snapshot
        self.memory_update(x)        
        self.gradients_allocated = True 
    
    def memory_update(self, x):
        """
            update snapshot and full gradient estimator stored in memory
        """
        # full_gradient = gradient F(snapshot)
        self._full_gradient(x, out = self.full_gradient)
        self.snapshot = x.clone()

    def memory_reset(self):
        """        
            resets subset gradients and full gradient in memory.
        """
        if self.gradients_allocated == True:
            del(self.subset_gradients)
            del(self.full_gradient)
            del(self.tmp2)
            del(self.tmp1)

            self.gradients_allocated = False


class LSVRGFunction(SVRGFunction):
    '''Class for use as objective function in gradient type algorithms to enable the use of subsets.
    The `gradient` method doesn't return the mathematical gradient of the sum of functions, 
    but rather a variance-reduced approximated gradient corresponding to the minibatch SVRG algorithm.
    More details can be found below, in the gradient method.
    Parameters:
    -----------
    - (optional) 
        precond: function taking into input an integer (subset_num) and a DataContainer and outputting a DataContainer
        serves as diagonal preconditioner
        default None
           
    '''
    def __init__(self, functions, precond=None, update_frequency = 2, **kwargs):

        super(LSVRGFunction, self).__init__(functions, precond=None, update_frequency = 2, **kwargs)
        self.probability_threshold = 1.0/(self.update_frequency*self.num_subsets)
        self.update_probability = 1.0 

    def gradient(self, x, out=None):
        """
        Returns a variance-reduced approximate gradient, defined below.
        For f = 1/num_subsets \sum_{i=1}^num_subsets f_i, the output is computed as follows:
            - choose a subset j with function next_subset()
            - check if full_grad_at_snapshot and snapshot should be updated
            - if they shouldn't then compute
                subset_gradient - subset_gradient_at_snapshot + full_grad_at_snapshot
                where 
                - subset_grad is the gradient of function number j at current_iterate
                - subset_grad_at_snapshot is the gradient of function number j at snapshot in memory
                - full_grad_at_snapshot is the approximation of the gradient of f in memory
                    computed as full_gradient = 1/num_subsets \sum_{i=1}^num_subsets subset_gradient_{i} (snapshot)
            - otherwise update full_grad_at_snapshot and snapshot
            - this gives an unbiased estimator of the gradient

        Reference:
        Kovalev, Dmitry; Horvath, Samuel; Richtarik, Peter
        "Don’t Jump Through Hoops and Remove Those Loops:
        SVRG and Katyusha are Better Without the Outer Loop." 
        International Conference on algorithmioc Learning Theory. 2020.
        """
        if not self.gradients_allocated:
            self.memory_init(x)         

        # Select the next subset (uniformly at random by default). This generates self.subset_num
        self.next_subset()

        # In first iteration and if the snapshot image has just been updated, the current iterate and snapshot will be the same
        # thus tmp2 = 0
        if self.iter == 0 or self.update_probability < self.probability_threshold:
            self.tmp2 = 0.0 * self.full_gradient
       # Otherwise, compute the difference between the subset gradient at current iterate and at the snapshot in memory    
        else:
            # Compute new gradient for current subset, store in tmp1
            # tmp1 = gradient F_{subset_num} (x)
            self.functions[self.subset_num].gradient(x, out=self.tmp1) 

            # Compute difference between current subset function gradient at current iterate (tmp1) and at snapshot, store in tmp2
            # tmp2 = gradient F_{subset_num} (x) - gradient F_{subset_num} (snapshot)
            self.tmp1.axpby(1., -1., self.functions[self.subset_num].gradient(self.snapshot), out=self.tmp2) 

        # Compute the output: tmp2 + full_grad
        if out is None:
            ret = 0.0 * self.tmp2
            self.tmp2.axpby(1., 1., self.full_gradient, out=ret)
        else:
            self.tmp2.axpby(1., 1., self.full_gradient, out=out)

        self.iter +=1 

        # Check whether to update the full gradient and snapshot image
        # Draw uniformly at random in [0, 1], if the value is below self.update_probability, then update
        # Setting self.update_frequency to inf indicates that we we never want to conduct the update
        self.update_probability = np.random.uniform() 
        if np.isinf(self.update_frequency) == False and self.update_probability < self.probability_threshold: 
            self.memory_update(x)

        # Apply preconditioning
        if self.precond is not None:
            if out is None:
                ret.multiply(self.precond(self.subset_num,x),out=ret)
            else:
                out.multiply(self.precond(self.subset_num,x),out=out)

        if out is None:
            return ret
