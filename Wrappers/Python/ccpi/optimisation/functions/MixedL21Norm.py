# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from ccpi.optimisation.functions import Function
from ccpi.framework import BlockDataContainer
import numpy as np


class MixedL21Norm(Function):
    
    
    """ MixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
    
    """      
    
    def __init__(self, **kwargs):

        super(MixedL21Norm, self).__init__()  
                    
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 

        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
              
        return x.pnorm(p=2).sum()                                
            
                            
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the MixedL21Norm function at x.
        
        This is the Indicator function of :math:`\mathbb{I}_{\{\|\cdot\|_{2,\infty}\leq1\}}(x^{*})`,
        
        i.e., 
        
        .. math:: \mathbb{I}_{\{\|\cdot\|_{2, \infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{if } \|x\|_{2, \infty}\leq1\\
            \infty, \mbox{otherwise}
            \end{cases}
        
        where, 
        
        .. math:: \|x\|_{2,\infty} = \max\{ \|x\|_{2} \} = \max\{ \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \dots}\}
        
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                                        
        tmp = (x.pnorm(2).max() - 1)
        if tmp<=1e-5:
            return 0
        else:
            return np.inf
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the MixedL21Norm function at x.
        
        .. math :: \mathrm{prox}_{\tau F}(x) = \frac{x}{\|x\|_{2}}\max\{ \|x\|_{2} - \tau, 0 \}
        
        where the convention 0 · (0/0) = 0 is used.
        
        """
        
        if out is None:
            
        
            tmp = x.pnorm(2)
            res = (tmp - tau).maximum(0.0) * x/tmp
            
            # TODO avoid using numpy, add operation in the framework
            # This will be useful when we add cupy 
                                 
            for el in res.containers:
                
                elarray = el.as_array()
                elarray[np.isnan(elarray)]=0
                el.fill(elarray)                
                
#                el.as_array()[np.isnan(el.as_array())]=0            
            
            return res            
            
        else:
            
#            tmp = x.pnorm(2)
#            res = (tmp - tau).maximum(0.0) * x/tmp
#
#            for el in res.containers:
#                
#                elarray = el.as_array()
#                elarray[np.isnan(elarray)]=0
#                el.fill(elarray)  
#
#            out.fill(res)   
            
            
            tmp = x.pnorm(2)
            tmp_ig = 0.0 * tmp
            (tmp - tau).maximum(0.0, out = tmp_ig)
            tmp_ig.multiply(x, out = out)
            out.divide(tmp, out = out)
            
            for el in out.containers:
                
                elarray = el.as_array()
                elarray[np.isnan(elarray)]=0
                el.fill(elarray)  

            out.fill(out)              
            
            
        
# TODO, add the prox conjugate in the documenataion and then delete the code below
            
##############################################################################
##############################################################################
#    def proximal_conjugate(self, x, tau, out=None): 
#
#        
#        if out is None:                                        
#            tmp = x.get_item(0) * 0	
#            for el in x.containers:	
#                tmp += el.power(2.)	
#            tmp.sqrt(out=tmp)	
#            tmp.maximum(1.0, out=tmp)	
#            frac = [ el.divide(tmp) for el in x.containers ]	
#            return BlockDataContainer(*frac)
#        
#    
#        else:
#                            
#            res1 = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 )
#            res1.sqrt(out=res1)	
#            res1.maximum(1.0, out=res1)	
#            x.divide(res1, out=out)
                              
                
class SmoothMixedL21Norm(Function):
    
    """ SmoothMixedL21Norm function: :math:`F(x) = ||x||_{2,1} = \sum |x|_{2} = \sum \sqrt{ (x^{1})^{2} + (x^{2})^{2} + \epsilon^2 + \dots}`                  
    
        where x is a BlockDataContainer, i.e., :math:`x=(x^{1}, x^{2}, \dots)`
        
        Conjugate, proximal and proximal conjugate methods no closed-form solution
        
    
    """    
        
    def __init__(self, epsilon):
                
        r'''
        :param epsilon: smoothing parameter making MixedL21Norm differentiable 
        '''

        super(SmoothMixedL21Norm, self).__init__(L=1)          
        self.epsilon = epsilon   
                
        if self.epsilon==0:
            raise ValueError('We need epsilon>0. Otherwise, call "MixedL21Norm" ')
                            
    def __call__(self, x):
        
        r"""Returns the value of the SmoothMixedL21Norm function at x.                                            
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
            
            
        return (x.pnorm(2)**2 + self.epsilon**2).sqrt().sum()
         

    def gradient(self, x, out=None): 
        
        r"""Returns the value of the gradient of the SmoothMixedL21Norm function at x.
        
        \frac{x}{|x|}
                
                
        """     
        
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
                   
        denom = (x.pnorm(2)**2 + self.epsilon**2).sqrt()
                          
        if out is None:
            return x.divide(denom)
        else:
            x.divide(denom, out=out)        

if __name__ == '__main__':
    
# TODO delete test below    
    
    M, N, K = 2,3,50
    from ccpi.framework import BlockGeometry, ImageGeometry
    import numpy
    
    ig = ImageGeometry(M, N)
    
    BG = BlockGeometry(ig, ig)
    
    U = BG.allocate('random')
    
    # Define no scale and scaled
    alpha = 0.5
    f_no_scaled = MixedL21Norm() 
    f_scaled = alpha * MixedL21Norm()  
    
    # call
    
    a1 = f_no_scaled(U)
    a2 = f_scaled(U)    
    print(a1, 2*a2)
        
    
    print( " ####### check without out ######### " )
          
          
    u_out_no_out = BG.allocate('random')         
    res_no_out = f_scaled.proximal_conjugate(u_out_no_out, 0.5)          
    print(res_no_out[0].as_array())
    
    print( " ####### check with out ######### " ) 
#          
    res_out = BG.allocate()        
    f_scaled.proximal_conjugate(u_out_no_out, 0.5, out = res_out)
#    
    print(res_out[0].as_array())   
#
    numpy.testing.assert_array_almost_equal(res_no_out[0].as_array(), \
                                            res_out[0].as_array(), decimal=4)

    numpy.testing.assert_array_almost_equal(res_no_out[1].as_array(), \
                                            res_out[1].as_array(), decimal=4)     
    
    
    tau = 0.4
    d1 = f_scaled.proximal(U, tau)
    
    tmp = (U.get_item(0)**2 + U.get_item(1)**2).sqrt()
    
    d2 = (tmp - alpha*tau).maximum(0) * U/tmp
    
    numpy.testing.assert_array_almost_equal(d1.get_item(0).as_array(), \
                                            d2.get_item(0).as_array(), decimal=4) 

    numpy.testing.assert_array_almost_equal(d1.get_item(1).as_array(), \
                                            d2.get_item(1).as_array(), decimal=4)     
    
    out1 = BG.allocate('random')
    
    
    f_scaled.proximal(U, tau, out = out1)
    
    numpy.testing.assert_array_almost_equal(out1.get_item(0).as_array(), \
                                            d1.get_item(0).as_array(), decimal=4) 

    numpy.testing.assert_array_almost_equal(out1.get_item(1).as_array(), \
                                            d1.get_item(1).as_array(), decimal=4)   
    
    f_scaled.proximal_conjugate(U, tau, out = out1)
    x = U
    tmp = x.get_item(0) * 0	
    for el in x.containers:	
        tmp += el.power(2.)	
    tmp.sqrt(out=tmp)	
    (tmp/f_scaled.scalar).maximum(1.0, out=tmp)	
    frac = [ el.divide(tmp) for el in x.containers ]	
    out2 = BlockDataContainer(*frac)   
    
    numpy.testing.assert_array_almost_equal(out1.get_item(0).as_array(), \
                                            out2.get_item(0).as_array(), decimal=4) 
    
    
    # check convex conjugate
    
    f = MixedL21Norm()
    x = BG.allocate('random')
    
    res1 = f.convex_conjugate(x)
    tmp = (x.pnorm(2).max() - 1)
    if tmp<=1e-5:
        res2=0
    else:
        res2=np.inf
    numpy.testing.assert_almost_equal(res1, res2) 
    print("Convex conjugate is .... OK")
    
    
    ig = ImageGeometry(4, 5)
    bg = BlockGeometry(ig, ig)
    
    epsilon = 0.5
    
    f1 = SmoothMixedL21Norm(epsilon)    
    x = bg.allocate('random')
    
    
    print("Check call for smooth MixedL21Norm ...OK")
    
    # check call
    res1 = f1(x)
    
    
    res2 = (x.pnorm(2)**2 + epsilon**2).sqrt().sum()
#    tmp = x.get_item(0) * 0.
#    for el in x.containers:
#        tmp += el.power(2.)
#    tmp+=epsilon**2        
#    res2 = tmp.sqrt().sum()

    # alternative        
    tmp1 = x.copy()
    tmp1.containers += (epsilon,)        
    res3 = tmp1.pnorm(2).sum()
                    
    numpy.testing.assert_almost_equal(res1, res2, decimal=4) 
    numpy.testing.assert_almost_equal(res1, res3, decimal=4) 
    
    print("Check gradient for smooth MixedL21Norm ... OK ")        
    
    res1 = f1.gradient(x)
    res2 = x.divide((x.pnorm(2)**2 + epsilon**2).sqrt())
    numpy.testing.assert_array_almost_equal(res1.get_item(0).as_array(), 
                                            res2.get_item(0).as_array()) 
    
    numpy.testing.assert_array_almost_equal(res1.get_item(1).as_array(), 
                                            res2.get_item(1).as_array()) 
    
    # check with MixedL21Norm, when epsilon close to 0
    
    print("Check as epsilon goes to 0 ... OK") 
    
    f1 = SmoothMixedL21Norm(1e-12)   
    f2 = MixedL21Norm()
    
    res1 = f1(x)
    res2 = f2(x)
    numpy.testing.assert_almost_equal(f1(x), f2(x))     




    
    
    
    
    
    
    
    
    

      
    
    

      
    
    
    

    

    
