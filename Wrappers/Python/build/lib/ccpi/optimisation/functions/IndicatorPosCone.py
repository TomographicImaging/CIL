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
from ccpi.framework import ImageData

class IndicatorBox(Function):
    '''Box constraints indicator function. 
    
    Calling returns 0 if argument is within the box. The prox operator is projection onto the box. 
    Only implements one scalar lower and one upper as constraint on all elements. Should generalise
    to vectors to allow different constraints one elements.
'''
    
    def __init__(self,lower=0,upper=numpy.inf):
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
        # support function sup <x, z>, z \in [lower, upper]
        # ????
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

    from ccpi.framework import ImageGeometry, BlockDataContainer

    N, M = 2,3
    ig = ImageGeometry(voxel_num_x = N, voxel_num_y = M)            
    
    u = ig.allocate('random_int')
    tau = 2
    
    f = IndicatorBox(2, 3)
    
    lower = 10
    upper = 30
        
    z1 = f.proximal(u, tau)
    
    z2 = f.proximal_conjugate(u/tau, 1/tau)
    
    z = z1 + tau * z2
    
    numpy.testing.assert_array_equal(z.as_array(), u.as_array())  

    out1 = ig.allocate()
    out2 = ig.allocate()
    
    f.proximal(u, tau, out=out1)
    f.proximal_conjugate(u/tau, 1/tau, out = out2)
    
    p = out1 + tau * out2
    
    numpy.testing.assert_array_equal(p.as_array(), u.as_array()) 
    
    d = f.convex_conjugate(u)
    print(d)
    
    
    
    # what about n-dimensional Block
    #uB = BlockDataContainer(u,u,u)
    #lowerB = BlockDataContainer(1,2,3)
    #upperB = BlockDataContainer(10,21,30)
    
    #fB = IndicatorBox(lowerB, upperB)
    
    #z1B = fB.proximal(uB, tau)
    
    
    
    
    
    
    
    