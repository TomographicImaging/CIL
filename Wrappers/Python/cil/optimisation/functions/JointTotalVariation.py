from cil.optimisation.functions import Function
from cil.optimisation.operators import GradientOperator
from cil.framework import BlockDataContainer, BlockGeometry

import numpy as np

class SmoothJointTV(Function):
              
    def __init__(self, eta, axis, lambda_par, correlation="Space", backend="c"):
                
        r'''

        :param eta: Smoothing parameter that allows to differentiate SmoothJointTV 
        :type eta: non-zero integer
        :param axis: Differentiation will be applied with respect to axis
        :type axis: int
        :param lambda_par: Regularisation parameter
      
        '''

        super(SmoothJointTV, self).__init__(L=8)
        
        # smoothing parameter
        self.eta = eta   

        if self.eta==0:
            raise ValueError('For SmoothJointTV a positive epsilon is expected, {} is passed.'.format(self.epsilon))

        ###########################################################################################                    
        # In CIL, we do not use a "domain" when a function is defined. Therefore, we need to acquire
        # the image domain, which then will be used by the GradientOperator.  
        ###########################################################################################

        self._gradient_operator = None
        self._domain = None
        
        # Select axis to differentiate
        self.axis = axis
        
        # Regularisation parameter    
        self.lambda_par=lambda_par    

        # correlation space or spacechannels
        self.correlation = correlation
        self.backend = backend         
                                    
                            
    def __call__(self, x):
        
        r""" x is BlockDataContainer that contains 2 DataContainers, i.e., (u,v).
        """

        # We need a domain in order to define our GradientOperator. We extract the geometry from one of 
        # the container.

        self._domain = x[0].geometry

        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 

        tmp = np.abs((self.lambda_par*self.gradient_operator.direct(x[0]).pnorm(2).power(2) +\
             (1-self.lambda_par)*self.gradient_operator.direct(x[1]).pnorm(2).power(2)+\
              self.eta**2).sqrt().sum())

        return tmp    
                                     
    def gradient(self, x, out=None):
        
        denom = (self.lambda_par*self.grad.direct(x[0]).pnorm(2).power(2) + (1-self.lambda_par)*self.grad.direct(x[1]).pnorm(2).power(2)+\
              self.eta**2).sqrt()         
        
        if self.axis==0:            
            num = self.lambda_par*self.grad.direct(x[0])                        
        else:            
            num = (1-self.lambda_par)*self.grad.direct(x[1])            

        if out is None:    
            tmp = self.grad.range.allocate()
            tmp[self.axis].fill(self.grad.adjoint(num.divide(denom)))
            return tmp
        else:                                
            self.grad.adjoint(num.divide(denom), out=out[self.axis])

    @property
    def gradient_operator(self):

        '''creates a gradient operator if not instantiated yet
        There is no check that the variable _domain is changed after instantiation (should not be the case)'''
        if self._gradient_operator is None:
            if self._domain is not None:
                self._gradient_operator = GradientOperator(self._domain, correlation = self.correlation, backend = self.backend)
        return self._gradient_operator 
       

