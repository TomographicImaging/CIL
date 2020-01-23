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
        '''Creator
        
        :param b:  translation of the function
        :type b: :code:`DataContainer`, optional
        '''
        super(MixedL21Norm, self).__init__()  
        self.b = kwargs.get('b', None)  

        # This is to handle tensor for Total Generalised Variation                    
        self.SymTensor = kwargs.get('SymTensor',False)
        
        # We use this parameter to make MixedL21Norm differentiable
#        self.epsilon = epsilon
        
        if self.b is not None and not isinstance(self.b, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(self.b)))
            
        
    def __call__(self, x):
        
        r"""Returns the value of the MixedL21Norm function at x. 

        :param x: :code:`BlockDataContainer`                                           
        """
        if not isinstance(x, BlockDataContainer):
            raise ValueError('__call__ expected BlockDataContainer, got {}'.format(type(x))) 
            
#        tmp_cont = x.containers                        
#        tmp = x.get_item(0) * 0.
#        for el in tmp_cont:
#            tmp += el.power(2.)
#        tmp.add(self.epsilon**2, out = tmp)    
#        return tmp.sqrt().sum()            
         
        y = x
        if self.b is not None:
            y = x - self.b    
        return y.pnorm(p=2).sum()                                
            
#        tmp = x.get_item(0) * 0.
#        for el in x.containers:
#            tmp += el.power(2.)
#        #tmp.add(self.epsilon, out = tmp)    
#        return tmp.sqrt().sum()

                            
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
                
#        tmp1 = x.get_item(0) * 0.
#        for el in x.containers:
#            tmp1 += el.power(2.)
#        tmp1.add(self.epsilon**2, out = tmp1)
#        tmp = tmp1.sqrt().as_array().max() - 1
#                    
#        if tmp<=1e-6:
#            return 0
#        else:
#            return np.inf            
                
        tmp = (x.pnorm(2).as_array().max() - 1)
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
            
            for el in res.containers:
                el.as_array()[np.isnan(el.as_array())]=0            
            
            return res            
            
        else:
            
            tmp = x.pnorm(2)
            res = (tmp - tau).maximum(0.0) * x/tmp

            for el in res.containers:
                el.as_array()[np.isnan(el.as_array())]=0

            out.fill(res)            
            
                        
#            tmp = functools.reduce(lambda a,b: a + b*b, x.containers, x.get_item(0) * 0 ).sqrt()
#            res = (tmp - tau).maximum(0.0) * x/tmp
#
#            for el in res.containers:
#                el.as_array()[np.isnan(el.as_array())]=0
#
#            out.fill(res)
        
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
                              

if __name__ == '__main__':
    
    M, N, K = 2,3,50
    from ccpi.framework import BlockGeometry, ImageGeometry
    import numpy
    
    ig = ImageGeometry(M, N)
    
    BG = BlockGeometry(ig, ig)
    
    U = BG.allocate('random_int')
    
    # Define no scale and scaled
    alpha = 0.5
    f_no_scaled = MixedL21Norm() 
    f_scaled = alpha * MixedL21Norm()  
    
    # call
    
    a1 = f_no_scaled(U)
    a2 = f_scaled(U)    
    print(a1, 2*a2)
        
    
    print( " ####### check without out ######### " )
          
          
    u_out_no_out = BG.allocate('random_int')         
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
    
    out1 = BG.allocate('random_int')
    
    
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
    
    

      
    
    
    

    

    