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
from ccpi.framework import ImageData, ImageGeometry
import functools

class KullbackLeibler(Function):
    
    ''' Assume that data > 0
                
    '''
    
    def __init__(self,data, **kwargs):
        
        super(KullbackLeibler, self).__init__()
        
        self.b = data        
        self.bnoise = kwargs.get('bnoise', 0)

                                                
    def __call__(self, x):
                
        res_tmp = numpy.zeros(x.shape)
        
        tmp = x + self.bnoise 
        ind = x.as_array()>0
        
        res_tmp[ind] = x.as_array()[ind] - self.b.as_array()[ind] * numpy.log(tmp.as_array()[ind])
        
        return res_tmp.sum()

    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0,
                                datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        #TODO Division check
        if out is None:
            return 1 - self.b/(x + self.bnoise)
        else:
            x.add(self.bnoise, out=out)
            self.b.divide(out, out=out)
            out.subtract(1, out=out)
            out *= -1
            
    def convex_conjugate(self, x):
        
        tmp = self.b/(1-x)
        ind = tmp.as_array()>0
        
        return (self.b.as_array()[ind] * (numpy.log(tmp.as_array()[ind])-1)).sum()

    
    def proximal(self, x, tau, out=None):
        
        if out is None:        
            return 0.5 *( (x - self.bnoise - tau) + ( (x + self.bnoise - tau)**2 + 4*tau*self.b   ) .sqrt() )
        else:
            
            tmp =  0.5 *( (x - self.bnoise - tau) + 
                        ( (x + self.bnoise - tau)**2 + 4*tau*self.b   ) .sqrt()
                        )
            x.add(self.bnoise, out=out)
            out -= tau
            out *= out
            tmp = self.b * (4 * tau)
            out.add(tmp, out=out)
            out.sqrt(out=out)
            
            x.subtract(self.bnoise, out=tmp)
            tmp -= tau
            
            out += tmp
            
            out *= 0.5
                            
    def proximal_conjugate(self, x, tau, out=None):

                
        if out is None:
            z = x + tau * self.bnoise
            return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
        else:
#            z = x + tau * self.bnoise
#            out.fill( 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt()) )
            
            
            tmp1 = x + tau * self.bnoise - 1
            tmp2 = tmp1 + 2
            
            self.b.multiply(4*tau, out=out)       
            tmp1.multiply(tmp1, out=tmp1)
            out += tmp1
            out.sqrt(out=out)
                        
            out *= -1
            out += tmp2
            out *= 0.5

        
    
    def __rmul__(self, scalar):
        
        ''' Multiplication of L2NormSquared with a scalar
        
        Returns: ScaledFunction
                        
        '''
        
        return ScaledFunction(self, scalar)     
        
        
    

if __name__ == '__main__':
   
    
    from ccpi.framework import ImageGeometry
    import numpy
    
    N, M = 2,3
    ig  = ImageGeometry(N, M)
    data = ImageData(numpy.random.randint(-10, 10, size=(M, N)))
    x = ImageData(numpy.random.randint(-10, 10, size=(M, N)))
    
    bnoise = ImageData(numpy.random.randint(-10, 10, size=(M, N)))
    
    f = KullbackLeibler(data)

    print(f(x))
    
#    numpy.random.seed(10)
#    
#    
#    x = numpy.random.randint(-10, 10, size = (2,3))
#    b = numpy.random.randint(1, 10, size = (2,3))
#    
#    ind1 = x>0
#        
#    res = x[ind1] - b * numpy.log(x[ind1])
#    
##    ind = x>0
#    
##    y = x[ind]
#    
#    
#    
#    
#    

    
        
