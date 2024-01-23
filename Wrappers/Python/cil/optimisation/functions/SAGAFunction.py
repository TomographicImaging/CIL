from cil.optimisation.functions import SAGFunction

class SAGAFunction(SAGFunction):

    r""" Stochastic Average Gradient Ameliore (SAGA) Function

    The SAGAFunction represents the objective function :math:`\frac{1}{n}\sum_{i=1}^{n}F_{i}(x)`, where
    :math:`n` denotes the number of subsets. The :func:`~SAGFunction.gradient` corresponds to a 
    variance-reduced approximated gradient. More details can be found in :cite:`Defazio_et_al_2014`.

    Parameters:
    -----------
    functions : list(functions) 
                A list of functions: :code:`[F_{1}, F_{2}, ..., F_{n}]`. Each function is assumed to be smooth function with an implemented :func:`~Function.gradient` method.
    sampling : :obj:`string`, Default = :code:`random`
               Selection process for each function in the list. It can be :code:`random` or :code:`sequential`. 
    replacement : :obj:`boolean`. Default = :code:`True`
               The same subset can be selected when :code:`replacement=True`. 
    precond : DataContainer
               A preconditioner, i.e, an array that multiplies the output from the gradient of the selected function :math:`\partial_F_{i}(x)`.
    gradient_initial_point : DataContainer
               Initialize the subset gradients from initial point. Default = None and the initial point is 0.

    Note
    ----
        
    The :meth:`~SAGAFunction.gradient` computes the `gradient` of one function from the list :math:`[F_{1},\cdots,F_{n}]`,
    
    .. math:: \partial F_{i}(x) .

    The ith function is selected from the :meth:`~SubsetSumFunction.next_subset` method.

    Theoretical values

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
                
    
    """

    def __init__(self, functions, sampling = "random", precond=None, replacement = False, gradient_initial_point=None):
  
        self.gradient_initial_point = gradient_initial_point
        self.allocate_memory = False
        super(SAGFunction, self).__init__(functions, sampling = sampling, replacement=replacement)

    def gradient(self, x, out):
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

        # Allocate in memory a) subset_gradients, b) full_gradient and c) tmp1, tmp2
        if not self.allocate_memory:
            self.initialise_memory(x) 

        # Select the next subset 
        self.next_subset()

        # Compute gradient for current subset and store in tmp1
        self.functions[self.subset_num].gradient(x, out=self.tmp1)
        # Update the number of (statistical) passes over the entire data until this iteration 
        self.data_passes.append(self.data_passes[-1]+1./self.num_subsets)

        # Compute the difference between the gradient of subset_num function 
        # at current iterate and the subset gradient, which is stored in tmp2.
        # tmp2 = gradient F_{subset_num} (x) - subset_gradients_{subset_num}
        self.tmp1.sapyb(1., self.subset_gradients[self.subset_num], -1., out=self.tmp2)

        # Compute the output : tmp2 + full_gradient
        self.tmp2.sapyb(self.num_subsets, self.full_gradient, 1., out=out)

        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.subset_num].fill(self.tmp1)

        # Update the full gradient estimator: add (gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}) to the current full_gradient
        self.full_gradient.sapyb(1., self.tmp2, 1., out=self.full_gradient)
       

    