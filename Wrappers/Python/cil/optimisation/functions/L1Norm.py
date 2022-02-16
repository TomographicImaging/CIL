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
import numpy as np

 
class L1Norm(Function):
    
    r"""L1Norm Function

        The following cases are considered:   

        *  :math:`F(\cdot) = ||\cdot||_{1}`

        *  :math:`F(\cdot) = ||\cdot - \,b||_{1}`

        **kwargs
            b : DataContainer, default = None      
                Translates the function at point :code:`b`.

        Examples
        --------

        >>> F = L1Norm() # ( no data )
        >>> F = L1Norm(b=data) # ( with data )
    
                    
    """   
           
    def __init__(self, **kwargs):

        super(L1Norm, self).__init__()
        self.b = kwargs.get('b',None)
        
    def __call__(self, x):
        
        r"""Returns the value of the L1Norm function at :code:`x`.
        
        The following cases are considered:   

        *  :math:`F(x) = ||x||_{1}`

        *  :math:`F(x) = ||x - b||_{1}`      
        
        """
        
        y = x
        if self.b is not None: 
            y = x - self.b
        return y.abs().sum()  
          
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the L1Norm function at :code:`x`.

        The following cases are considered:  

        *  :math:`F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*})`

        *  :math:`F^{*}(x^{*}) = \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) \,+ <x^{*},b>`

        Note
        ----

        The convex conjugate of the L1Norm function, is the Indicator function of the unit ball of the
        :math:`L^{\infty}` norm: 

        .. math:: \mathbb{I}_{\{\|\cdot\|_{\infty}\leq1\}}(x^{*}) 
            = \begin{cases} 
            0, \mbox{ if } \|x^{*}\|_{\infty}\leq1\\
            \infty, \mbox{ otherwise }
            \end{cases}

        If :code:`b is not None`, the same formula as the convex conjugate of :py:meth:`TranslateFunction.convex_conjugate` is used.
    
        """        
        
        tmp = x.abs().max() - 1
        if tmp<=1e-5:            
            if self.b is not None:
                return self.b.dot(x)
            else:
                return 0.
        return np.inf    

    def _soft_shrinkage(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the function :math: `F(x) = \|x\|_{1}` at :code:'x'. 
        Also referred as soft-shrinkage operator.

        .. math:: \mathrm{prox}_{\tau \|\cdot\|_{1}}(x) = \mathrm{soft}(x, \tau) = ( |x| - \tau )_{+} \mathrm{sign}(x)

        """
        
        if out is None:
            return x.sign() * (x.abs() - tau).maximum(0) 
        else:
            x.abs(out = out)
            out -= tau
            out.maximum(0, out = out)
            out *= x.sign()  
                    
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the L1Norm function at :code:`x`.

        The following cases are considered:  

        *  :math:`\mathrm{prox}_{\tau F}(x) = \mathrm{soft}(x, \tau)`

        *  :math:`\mathrm{prox}_{\tau F}(x) = \mathrm{soft}(x - b, \tau) + b`
                
        where, :math:`\mathrm{soft}(x, \tau) := ( |x| - \tau )_{+} \mathrm{sign}(x)\,.`

        Note
        ----
        If :code:`b is not None`, the same formula as the proximal operator of :py:meth:`TranslateFunction.proximal` is used.
                             
        """  

                    
        if out is None:                                                
            if self.b is not None:                                
                return self.b + self._soft_shrinkage(x - self.b, tau)
            else:
                return self._soft_shrinkage(x, tau)             
        else: 
            
            if self.b is not None:
                self._soft_shrinkage(x - self.b, tau, out = out)
                out += self.b
            else:
                self._soft_shrinkage(x, tau, out = out)       

