from cil.optimisation.functions import ApproximateGradientSumFunction

class SAGFunction(ApproximateGradientSumFunction):

    r""" Stochastic Average Gradient (SAG) Function

    # TODO Improve doc

    """

    def __init__(self, functions, selection=None, gradient_initial_point=None):
 
        self.gradient_initial_point = gradient_initial_point
        self.allocate_memory = False
        super(SAGFunction, self).__init__(functions, selection = selection)

    def approximate_gradient(self, function_num, x, out):

        """
        # TODO Improve doc: Returns a variance-reduced approximate gradient.        
        """

        # Allocate in memory a) subset_gradients, b) full_gradient and c) tmp1, tmp2
        if not self.allocate_memory:
            self.initialise_memory(x) 

        # Compute gradient for current subset and store in tmp1
        self.functions[self.function_num].gradient(x, out=self.tmp1)

        # Compute the difference between the gradient of subset_num function 
        # at current iterate and the subset gradient, which is stored in tmp2.
        # tmp2 = gradient F_{subset_num} (x) - subset_gradients_{subset_num}        
        self.tmp1.sapyb(1., self.subset_gradients[self.function_num], -1., out=self.tmp2)

        # Compute the output : tmp2 + full_gradient
        self.tmp2.sapyb(self.num_functions , self.full_gradient, 1., out=out)

        # Update subset gradients in memory: store the computed gradient F_{subset_num} (x) in self.subset_gradients[self.subset_num]
        self.subset_gradients[self.function_num].fill(self.tmp1)

        # Update the full gradient estimator: add (gradient F_{subset_num} (x) - subset_gradient_in_memory_{subset_num}) to the current full_gradient
        self.full_gradient.sapyb(1., self.tmp2, 1., out=self.full_gradient)

    def initialise_memory(self, x):

        r"""Initialize subset gradients :math:`v_{i}` and full gradient that are stored in memory.
        The initial point is 0 by default.
        """
        
        # Default initialisation point = 0
        if self.gradient_initial_point is None:
            self.subset_gradients = [ x * 0.0 for _ in range(self.num_functions )]
            self.full_gradient = x * 0.0
        # Otherwise, initialise subset gradients in memory and the full gradient at the provided gradient_initialisation_point
        else:
            self.subset_gradients = [ fi.gradient(self.gradient_initial_point) for i, fi in enumerate(self.functions)]
            self.full_gradient =  sum(self.subset_gradients)

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