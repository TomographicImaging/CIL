#========================================================================
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#=========================================================================

from ccpi.optimisation.functions import Function
import warnings

# Define a class for squared 2-norm
class Norm2Sq(Function):
    '''
    f(x) = c*||A*x-b||_2^2
    
    which has 
    
    grad[f](x) = 2*c*A^T*(A*x-b)
    
    and Lipschitz constant
    
    L = 2*c*||A||_2^2 = 2*s1(A)^2
    
    where s1(A) is the largest singular value of A.
    
    '''
    
    def __init__(self,A,b,c=1.0,memopt=False):
        super(Norm2Sq, self).__init__()
    
        self.A = A  # Should be an operator, default identity
        self.b = b  # Default zero DataSet?
        self.c = c  # Default 1.
        if memopt:
            try:
                self.range_tmp = A.range_geometry().allocate()
                self.domain_tmp = A.domain_geometry().allocate()
                self.memopt = True
            except NameError as ne:
                warnings.warn(str(ne))
                self.memopt = False
            except NotImplementedError as nie:
                print (nie)
                warnings.warn(str(nie))
                self.memopt = False
        else:
            self.memopt = False
        
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        try:
            self.L = 2.0*self.c*(self.A.norm()**2)
        except AttributeError as ae:
            pass
        except NotImplementedError as noe:
            pass
        
    #def grad(self,x):
    #    return self.gradient(x, out=None)

    def __call__(self,x):
        #return self.c* np.sum(np.square((self.A.direct(x) - self.b).ravel()))
        #if out is None:
        #    return self.c*( ( (self.A.direct(x)-self.b)**2).sum() )
        #else:
        y = self.A.direct(x)
        y.__isub__(self.b)
        #y.__imul__(y)
        #return y.sum() * self.c
        try:
            return y.squared_norm() * self.c
        except AttributeError as ae:
            # added for compatibility with SIRF 
            return (y.norm()**2) * self.c
    
    def gradient(self, x, out = None):
        if self.memopt:
            #return 2.0*self.c*self.A.adjoint( self.A.direct(x) - self.b )
            self.A.direct(x, out=self.range_tmp)
            self.range_tmp -= self.b 
            self.A.adjoint(self.range_tmp, out=out)
            #self.direct_placehold.multiply(2.0*self.c, out=out)
            out *= (self.c * 2.0)
        else:
            return (2.0*self.c)*self.A.adjoint( self.A.direct(x) - self.b )
