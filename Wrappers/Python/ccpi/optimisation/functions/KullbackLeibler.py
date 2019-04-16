# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library developed by
#   Visual Analytics and Imaging System Group of the Science Technology
#   Facilities Council, STFC

#   Copyright 2018-2019 Evangelos Papoutsellis and Edoardo Pasca

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#       http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

import numpy
from ccpi.optimisation.functions import Function
from ccpi.optimisation.functions.ScaledFunction import ScaledFunction
from ccpi.framework import DataContainer, ImageData, ImageGeometry  

class KullbackLeibler(Function):
    
    def __init__(self,data,**kwargs):
        
        super(KullbackLeibler, self).__init__()
        
        self.b = data        
        self.bnoise = kwargs.get('bnoise', 0)

        self.sum_value = self.b + self.bnoise        
        if  (self.sum_value.as_array()<0).any():
            self.sum_value = numpy.inf
                                                
    def __call__(self, x):
        
        if self.sum_value==numpy.inf:
            return numpy.inf
        else:
            return numpy.sum( x.as_array() - self.b.as_array() * numpy.log(self.sum_value.as_array()))

        
    def gradient(self, x, out=None):
        if out is None:
            #TODO Division check
            return 1 - self.b/(x + self.bnoise)
        else:
            x.add(self.bnoise, out=out)
            self.b.divide(out, out=out)
            out *= -1
            out += 1
    
    def convex_conjugate(self, x, out=None):
        pass
    
    def proximal(self, x, tau, out=None):
        if out is None:
            z = x + tau * self.bnoise
            return (z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt()
        else:
            z_m = x + tau * self.bnoise - 1
            self.b.multiply(4*tau, out=out)
            z_m.multiply(z_m, out=z_m)
            out += z_m
            out.sqrt(out=out)
            # z = z_m + 2
            z_m.sqrt(out=z_m)
            z_m += 2
            out *= -1
            out += z_m
                
    def proximal_conjugate(self, x, tau, out=None):
        pass
        
        
    

if __name__ == '__main__':
    
    N, M = 2,3
    ig  = ImageGeometry(N, M)
    #data = ImageData(numpy.random.randint(-10, 100, size=(M, N)))
    #x = ImageData(numpy.random.randint(-10, 100, size=(M, N)))
    #bnoise = ImageData(numpy.random.randint(-100, 100, size=(M, N)))
    data = ig.allocate(ImageGeometry.RANDOM_INT)
    x = ig.allocate(ImageGeometry.RANDOM_INT)
    bnoise = ig.allocate(ImageGeometry.RANDOM_INT)
    
    out = ig.allocate()
    
    f = KullbackLeibler(data, bnoise=bnoise)
    print(f.sum_value)
    
    print(f(x))
    grad = f.gradient(x)
    f.gradient(x, out=out)
    numpy.testing.assert_array_equal(grad.as_array(), out.as_array())
    
    prox = f.proximal(x,1.2)
    #print(grad.as_array())
    f.proximal(x, 1.2, out=out)
    numpy.testing.assert_array_equal(prox.as_array(), out.as_array())
    


    
        