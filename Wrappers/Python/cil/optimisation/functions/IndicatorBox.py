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
import numpy

class IndicatorBox(Function):
    
    
    r"""IndicatorBox Function with constraints


    Parameters
    ----------

    lower : Real number, default = :code:`-numpy.inf`
            Lower bound 
    upper : Real number, default = :code:`numpy.inf`
            Upper bound

    .. math:: 
         
         F(x) = \mathbb{I}_{[a, b]} = \begin{cases}  
                                            0, \text{ if } x \in [a, b] \\
                                            \infty, \text{otherwise}
                                     \end{cases}

    Examples
    --------

    >>> from cil.optimisation.functions import IndicatorBox
    >>> F = IndicatorBox(lower=0.5, upper=10.0)
                                       
    
    References
    ----------

    `Characteristic function <https://en.wikipedia.org/wiki/Characteristic_function_(convex_analysis)>`_
     
    """
    
    def __init__(self,lower=-numpy.inf,upper=numpy.inf):

        super(IndicatorBox, self).__init__()
        self.lower = lower
        self.upper = upper

    def __call__(self,x):

        r"""Returns the value of the IndicatorBox function at :code:`x`.
        """     
                
        if (numpy.all(x.as_array() >= self.lower) and 
            numpy.all(x.as_array() <= self.upper) ):
            val = 0
        else:
            val = numpy.inf
        return val
    
    def gradient(self, x):

        """

        Raises
        -------
        ValueError 
            IndicatorBox function is not differentiable.

        """
        return ValueError('IndicatorBox function is not differentiable') 
    
    def convex_conjugate(self,x):
        
        r"""Returns the value of the convex conjugate of the IndicatorBox function at :code:`x`.

        .. math:: 
            
            f^{*}(x) = \mathbb{I}_{[a, b]} = \begin{cases}  
                                                0, \text{ if } x \in [a, b] \\
                                                \infty, \text{otherwise}
                                        \end{cases}            

        """

        return x.maximum(0).sum()
         
    def proximal(self, x, tau, out=None):

        r"""Returns the value of the proximal operator of the IndicatorBox function at :code:`x`. 
        
        :math:`prox_{\tau \,F}(x) = \min\{\max\{x, \mbox{lower}\}, \mbox{upper}\}`
                
        """
        
        if out is None:
            return (x.maximum(self.lower)).minimum(self.upper)        
        else:               
            x.maximum(self.lower, out=out)
            out.minimum(self.upper, out=out) 
            
