from cil.optimisation.functions import SubsetSumFunction

class SAGFunction(SubsetSumFunction):

    r""" Stochastic Average Gradient (SAG) Function

    The SAGFunction represents the objective function :math:`\frac{1}{n}\sum_{i=1}^{n}F_{i}(x)`, where
    :math:`n` denotes the number of subsets. The :func:`~SAGFunction.gradient` corresponds to a 
    variance-reduced approximated gradient. More details can be found in :cite:`Schmidt2016`.

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
        
    The :meth:`~SAGFunction.gradient` computes the `gradient` of one function from the list :math:`[F_{1},\cdots,F_{n}]`,
    
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

        self.precond = precond     
        self.gradient_initial_point = gradient_initial_point
        self.allocate_memory = False
        super(SAGFunction, self).__init__(functions, sampling = sampling, replacement=replacement)

    def gradient(self, x, out):

        """
        Returns a variance-reduced approximate gradient.        
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

        # Compute the output : 1/num_subsets * tmp2 + full_gradient
        self.tmp2.sapyb(1./self.num_subsets, self.full_gradient, 1., out=out)

        # Apply preconditioning
        if self.precond is not None:
            out.multiply(self.precond(self.subset_num, x), out=out) 

        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.subset_num].fill(self.tmp1)

        # Update the full gradient estimator: add 1/num_subsets * (gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}) to the current full_gradient
        self.full_gradient.sapyb(1., self.tmp2, 1./self.num_subsets, out=self.full_gradient)

    def initialise_memory(self, x):

        r"""Initialize subset gradients :math:`v_{i}` and full gradient that are stored in memory.
        The initial point is 0 by default.
        """
        
        # Default initialisation point = 0
        if self.gradient_initial_point is None:
            self.subset_gradients = [ x * 0.0 for _ in range(self.num_subsets)]
            self.full_gradient = x * 0.0
        # Otherwise, initialise subset gradients in memory and the full gradient at the provided gradient_initialisation_point
        else:
            self.subset_gradients = [ fi.gradient(self.gradient_initial_point) for i, fi in enumerate(self.functions)]
            self.full_gradient = 1/self.num_subsets * sum(self.subset_gradients)
            # Compute the number of (statistical) passes over the entire data until this iteration 
            self.data_passes.append(self.data_passes[-1]+1.)

        self.tmp1 = x * 0.0
        self.tmp2 = x * 0.0

        self.allocate_memory = True
    
    def reset_memory(self):
        """ Resets the memory from subset gradients and full gradient.
        """
        if self.allocate_memory == True:
            del(self.subset_gradients)
            del(self.full_gradient)
            del(self.tmp1)
            del(self.tmp2)

            self.allocate_memory = False