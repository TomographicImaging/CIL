# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from cil.optimisation.functions import Function
from cil.framework import DataContainer
from cil.optimisation.operators import DiagonalOperator

class L2NormSquared(Function):
    
    r""" L2NormSquared Function
          
    The following cases are considered:

    *  :math:`F(\cdot) = \|\cdot\|^{2}_{2}`

    *  :math:`F(\cdot) = \|\cdot - b\|^{2}_{2}`


    **kwargs
        b : DataContainer, default = None      
            Translates the function at point :code:`b`.

    Examples
    --------

    :math:`F(x) = \|x\|_{2}^{2}`

    :math:`F(x) = \|x - b\|_{2}^{2}`    

    >>> from cil.optimisation.functions import L2NormSquared 
    >>> from cil.framework import ImageGeometry   
    >>> F = L2NormSquared() # ( no data )
    >>> ig = ImageGeometry(3,4)
    >>> data = ig.allocate('random')
    >>> F = L2NormSquared(b=data) # ( with data )
        
                                                  
    """    
    
    def __init__(self, **kwargs):
                      
        super(L2NormSquared, self).__init__(L = 2)
        self.b = kwargs.get('b',None) 
        
                            
    def __call__(self, x):

        r"""Returns the value of the L2NormSquared function at :code:`x`.
        
        The following cases are considered:   

        *  :math:`F(x) = \|x\|^{2}_{2}`

        *  :math:`F(x) = \|x - b\|^{2}_{2}`      
        
        """        
                  
        y = x
        if self.b is not None: 
            y = x - self.b
        try:
            return y.squared_norm()
        except AttributeError as ae:
            # added for compatibility with SIRF 
            return (y.norm()**2)
                
    def gradient(self, x, out=None):        

        r"""Returns the value of the gradient of the L2NormSquared function at :code:`x`.

        The following cases are considered:  

        *  :math:`F'(x) = 2*x`

        *  :math:`F'(x) = 2*(x-b)`

        Note
        ----        
        If :code:`b is not None`, the same formula of the gradient of :py:meth:`TranslateFunction.gradient` is used.

        """
                
        if out is not None:
            
            out.fill(x)
            if self.b is not None:
                out -= self.b
            out *= 2
            
        else:
            
            y = x
            if self.b is not None:
                y = x - self.b
            return 2*y
        
                                                       
    def convex_conjugate(self, x):

        r"""Returns the value of the convex conjugate of the L2NormSquared function at :code:`x`.

        The following cases are considered:  

        *  :math:`F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2}`

        *  :math:`F^{*}(x^{*}) = \frac{1}{4}\|x^{*}\|^{2}_{2} + <x^{*}, b>`

        Note
        ----        
        If :code:`b is not None`, the same formula of the convex conjugate of :py:meth:`TranslateFunction.convex_conjugate` is used.

        """
              
        tmp = 0
        
        if self.b is not None:
            tmp = x.dot(self.b) 
            
        return (1./4.) * x.squared_norm() + tmp


    def proximal(self, x, tau, out = None):
        
        r"""Returns the value of the proximal operator of the L2NormSquared function at :code:`x`.
        
        
        The following cases are considered:  

        *  :math:`\mathrm{prox}_{\tau F}(x) = \frac{x}{1+2\tau}`  

        *  :math:`\mathrm{prox}_{\tau F}(x) = \frac{x-b}{1+2\tau} + b`
                
        Note
        ----        
        If :code:`b is not None`, the same formula of the proximal operator of :py:meth:`TranslateFunction.proximal` is used.

        """          

        if out is None:
            
            if self.b is None:
                return x/(1+2*tau)
            else:
                tmp = x.subtract(self.b)
                tmp /= (1+2*tau)
                tmp += self.b
                return tmp

        else:
            if self.b is not None:
                x.subtract(self.b, out=out)
                out /= (1+2*tau)
                out += self.b
            else:
                x.divide((1+2*tau), out=out)


class WeightedL2NormSquared(Function):
    
    r""" WeightedL2NormSquared function

    The following cases are considered:   

    *  :math:`F(\cdot) = \| \cdot\|_{w}^{2}_{2}`

    *  :math:`F(\cdot) = ||\cdot - \,b||_{w,2}^{2}`,

    where, :math:`\| x\|_{w}^{2}_{2} = \sum w * |x|^{2}`.

    **kwargs
        b : DataContainer, default = None      
            Translates the function at point :code:`b`.
        weight: DataContainer, default = 1.0
            Weight for the L2NormSquared

    Examples
    --------

    >>> from cil.optimisation.functions import WeightedL2NormSquared 
    >>> from cil.framework import ImageGeometry 
    >>> F = WeightedL2NormSquared(weight = 5.0) # ( no data )
    >>> ig = ImageGeometry(3,4)    
    >>> data = ig.allocate('random')    
    >>> F = WeightedL2NormSquared(b = data, weight = 2.0) # ( with data )             
                                                              
    """                 
              
    def __init__(self, **kwargs):
                                 
       self.weight = kwargs.get('weight', 1.0) 
       self.b = kwargs.get('b', None) 
       tmp_norm = 1.0  
       self.tmp_space = self.weight*0. 
                            
       if isinstance(self.weight, DataContainer):
           self.operator_weight = DiagonalOperator(self.weight) 
           tmp_norm = self.operator_weight.norm() 
           self.tmp_space = self.operator_weight.domain_geometry().allocate()  
           
           if (self.weight<0).any():
               raise ValueError('Weigth contains negative values')             

       super(WeightedL2NormSquared, self).__init__(L = 2 * tmp_norm  ) 
    
    def __call__(self, x):
       
       self.operator_weight.direct(x, out = self.tmp_space)
       y = x.dot(self.tmp_space)
       
       if self.b is not None:
           self.operator_weight.direct(x - self.b, out = self.tmp_space)
           y = (x - self.b).dot(self.tmp_space)
       return y
     
                
    def gradient(self, x, out=None):        
        
                
        if out is not None:
            
            out.fill(x)        
            if self.b is not None:
                out -= self.b
            self.operator_weight.direct(out, out=out)                            
            out *= 2
            
        else:
            
            y = x
            if self.b is not None:
                y = x - self.b
            return 2*self.weight*y
    
    def convex_conjugate(self, x):
                      
        tmp = 0
        
        if self.b is not None:
            tmp = x.dot(self.b) 
            
        return (1./4) * (x/self.weight.sqrt()).squared_norm() + tmp
    
    def proximal(self, x, tau, out = None):
                  

        if out is None:
            
            if self.b is None:
                return x/(1+2*tau*self.weight)
            else:
                tmp = x.subtract(self.b)
                tmp /= (1+2*tau*self.weight)
                tmp += self.b
                return tmp

        else:
            
            if self.b is not None:
                x.subtract(self.b, out=out)
                out /= (1+2*tau*self.weight)
                out += self.b
            else:
                x.divide((1+2*tau*self.weight), out=out)                                     

