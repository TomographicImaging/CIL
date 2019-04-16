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
from ccpi.framework import ImageData

class KullbackLeibler(Function):
    
    ''' Assume that data > 0
                
    '''
    
    def __init__(self,data, **kwargs):
        
        super(KullbackLeibler, self).__init__()
        
        self.b = data        
        self.bnoise = kwargs.get('bnoise', 0)

                                                
    def __call__(self, x):
        
        # TODO check
        
        self.sum_value = x + self.bnoise        
        if  (self.sum_value.as_array()<0).any():
            self.sum_value = numpy.inf
        
        if self.sum_value==numpy.inf:
            return numpy.inf
        else:
            tmp = self.sum_value.as_array()            
            return (x - self.b * ImageData( numpy.log(tmp))).sum()
            
#            return numpy.sum( x.as_array() - self.b.as_array() * numpy.log(self.sum_value.as_array()))

        
    def gradient(self, x, out=None):
        
        #TODO Division check
        if out is None:
            return 1 - self.b/(x + self.bnoise)
        else:
            self.b.divide(x+self.bnoise, out=out)
            out.subtract(1, out=out)
    
    def convex_conjugate(self, x):
        
        tmp = self.b.as_array()/( 1 - x.as_array() )
        
        return (self.b * ( ImageData( numpy.log(tmp) ) - 1 ) - self.bnoise * (x - 1)).sum()
#        return self.b * ( ImageData(numpy.log(self.b/(1-x)) - 1 )) - self.bnoise * (x - 1)
    
    def proximal(self, x, tau, out=None):
        
        if out is None:        
            return 0.5 *( (x - self.bnoise - tau) + ( (x + self.bnoise - tau)**2 + 4*tau*self.b   ) .sqrt() )
        else:
            tmp =  0.5 *( (x - self.bnoise - tau) + ( (x + self.bnoise - tau)**2 + 4*tau*self.b   ) .sqrt() )
            out.fill(tmp)
            
    
    def proximal_conjugate(self, x, tau, out=None):

                
        if out is None:
            z = x + tau * self.bnoise
            return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
        else:
            z = x + tau * self.bnoise
            res = 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
            out.fill(res)
            
        
    
    def __rmul__(self, scalar):
        
        ''' Multiplication of L2NormSquared with a scalar
        
        Returns: ScaledFunction
                        
        '''
        
        return ScaledFunction(self, scalar)     
        
        
    

if __name__ == '__main__':
    
    N, M = 2,3
    ig  = ImageGeometry(N, M)
    data = ImageData(numpy.random.randint(-10, 100, size=(M, N)))
    x = ImageData(numpy.random.randint(-10, 100, size=(M, N)))
    
    bnoise = ImageData(numpy.random.randint(-100, 100, size=(M, N)))
    
    f = KullbackLeibler(data, bnoise=bnoise)
    print(f.sum_value)
    
    print(f(x))


    
        