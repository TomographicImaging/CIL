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
import numpy

class IndicatorBox(Function):
    '''Box constraints indicator function. 
    
    Calling returns 0 if argument is within the box. The prox operator is projection onto the box. 
    Only implements one scalar lower and one upper as constraint on all elements. Should generalise
    to vectors to allow different constraints one elements.
'''
    
    def __init__(self,lower=-numpy.inf,upper=numpy.inf):
        # Do nothing
        super(IndicatorBox, self).__init__()
        self.lower = lower
        self.upper = upper
        
    
    def __call__(self,x):
        
        if (numpy.all(x.array>=self.lower) and 
            numpy.all(x.array <= self.upper) ):
            val = 0
        else:
            val = numpy.inf
        return val
    
    def gradient(self,x):
        return ValueError('Not Differentiable') 
    
    def convex_conjugate(self,x):
        # support function sup <x^*, x>
        return 0 
    
    def proximal(self, x, tau, out=None):
        
        if out is None:
            return (x.maximum(self.lower)).minimum(self.upper)        
        else:                   
            x.maximum(self.lower, out=out)
            out.minimum(self.upper, out=out) 
            
    def proximal_conjugate(self, x, tau, out=None):

        if out is None:
            
            return x - tau * self.proximal(x/tau, tau)
        
        else:
            
            self.proximal(x/tau, tau, out=out)
            out *= -1*tau
            out += x

            
            
if __name__ == '__main__':  

    from ccpi.framework import ImageGeometry

    N, M = 2,3
    ig = ImageGeometry(voxel_num_x = N, voxel_num_y = M)            
    
    u = ig.allocate('random_int')
    tau = 10
    
    f = IndicatorBox(2, 3)
    
    lower = 10
    upper = 30
    
    x = u
    
    z1 = (x.maximum(lower)).minimum(upper)
    
    z2 = x - tau * ((x/tau).maximum(lower)).minimum(upper)
    
    z = z1 + z2/tau
    
    print(z.array, x.array)
    
    
#    prox = f.proximal(u, tau)
#    prox_conj = f.proximal_conjugate(u/tau, tau) 
#    
#    
#    z = prox + tau * prox_conj
#    print(z.as_array(), u.array)
    
    
#    x - tau * ((x/tau).maximum(self.lower)).minimum(self.upper) + 
            
            
            


                
            
            
    
