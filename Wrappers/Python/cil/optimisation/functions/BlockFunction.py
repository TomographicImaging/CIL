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
from cil.framework import BlockDataContainer
from numbers import Number

class BlockFunction(Function):
    
    r""" BlockFunction 
    
    Represents a *separable sum* function defined as
    
    .. math:: F:X_{1}\times X_{2}\cdots\times X_{m} \rightarrow (-\infty, \infty]
    
    where :math:`F` is the separable sum of functions :math:`(f_{i})_{i=1}^{m}`,
    
    .. math:: F(x_{1}, x_{2}, \cdots, x_{m}) = \overset{m}{\underset{i=1}{\sum}}f_{i}(x_{i}), \mbox{ with } f_{i}: X_{i} \rightarrow (-\infty, \infty].

    Parameters
    ----------

    *functions : Function
                 Functions to set up a separable sum.
                 
    Note
    ----
    A nice property (due to it's separability structure) is that the proximal operator 
    can be decomposed along the proximal operators of each function :math:`f_{i}`.
    
    .. math:: \mathrm{prox}_{\tau F}(x) = ( \mathrm{prox}_{\tau f_{i}}(x_{i}) )_{i=1}^{m}
    
    In addition, if :math:`\tau := (\tau_{1},\dots,\tau_{m})`, then 
    
    .. math:: \mathrm{prox}_{\tau F}(x) = ( \mathrm{prox}_{\tau_{i} f_{i}}(x_{i}) )_{i=1}^{m}

    Examples
    --------
    :math:`F(x_{1}, x_{2}) = ||x_{1}||_{2}^{2} + ||x_{2}||_{1}`

    >>> from cil.optimisation.functions import BlockFunction, L1Norm, L2NormSquared
    >>> f1 = L2NormSquared()
    >>> f2 = L1Norm()
    >>> F = BlockFunction(f1, f2)

    See also
    --------
    :class:`.BlockOperator`      
    
    """
    
    def __init__(self, *functions):
                
        super(BlockFunction, self).__init__()
        self.functions = functions      
        self.length = len(self.functions)
       
    @property        
    def L(self):
        # compute Lipschitz constant if possible
        tmp_L = 0
        for func in self.functions:
            if func.L is not None:  
                tmp_L += func.L                
            else:
                tmp_L = None 
                break 
        return tmp_L     
                                
    def __call__(self, x):
        
        r""" Returns the value of the BlockFunction at :code:`x`.
        
        :math:`F(x) = \overset{m}{\underset{i=1}{\sum}}f_{i}(x_{i}), \mbox{ where } x = (x_{1}, x_{2}, \cdots, x_{m}), \quad i = 1,2,\dots,m .`

        Raises
        ------
        ValueError
            If the length of the BlockFunction is not the same as the length of the BlockDataContainer :code:`x`.
        
                            
        """
        
        if self.length != x.shape[0]:
            raise ValueError('BlockFunction and BlockDataContainer have incompatible size')
        t = 0                
        for i in range(x.shape[0]):
            t += self.functions[i](x.get_item(i))
        return t
    
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the BlockFunction at :math:`x`.       
        
        :math:`F^{*}(x^{*}) = \overset{m}{\underset{i=1}{\sum}}f_{i}^{*}(x^{*}_{i})`
            
  
        Raises
        ------
        ValueError
            If the length of the BlockFunction is not the same as the length of the BlockDataContainer :code:`x`.
        

        """     
        
        if self.length != x.shape[0]:
            raise ValueError('BlockFunction and BlockDataContainer have incompatible size')
        t = 0              
        for i in range(x.shape[0]):
            t += self.functions[i].convex_conjugate(x.get_item(i))
        return t  
    
    def proximal(self, x, tau, out = None):
        
        r"""Returns the value of the proximal operator of the BlockFunction at :code:`x`.
        
        :math:`\mathrm{prox}_{\tau F}(x) =  (\mathrm{prox}_{\tau f_{i}}(x_{i}))_{i=1}^{m}`
            
        Raises
        ------
        ValueError
            If the length of the BlockFunction is not the same as the length of the BlockDataContainer :code:`x`.
                        
        """
        
        if self.length != x.shape[0]:
            raise ValueError('BlockFunction and BlockDataContainer have incompatible size')        
        
        if out is None:
                
            out = [None]*self.length
            if isinstance(tau, Number):
                for i in range(self.length):
                    out[i] = self.functions[i].proximal(x.get_item(i), tau)
            else:
                for i in range(self.length):
                    out[i] = self.functions[i].proximal(x.get_item(i), tau.get_item(i))                        
            
            return BlockDataContainer(*out)  
        
        else:
            if isinstance(tau, Number):
                for i in range(self.length):
                    self.functions[i].proximal(x.get_item(i), tau, out[i])
            else:
                for i in range(self.length):
                    self.functions[i].proximal(x.get_item(i), tau.get_item(i), out[i])            
    
    def gradient(self, x, out=None):
        
        r"""Returns the value of the gradient of the BlockFunction function at :code:`x`.
        
        :math:`F'(x) = [f_{1}'(x_{1}), ... , f_{m}'(x_{m})]`
            
        Raises
        ------
        ValueError
            If the length of the BlockFunction is not the same as the length of the BlockDataContainer :code:`x`.
                           
        """        
        
        if self.length != x.shape[0]:
            raise ValueError('BlockFunction and BlockDataContainer have incompatible size')        
        
        out = [None]*self.length
        for i in range(self.length):
            out[i] = self.functions[i].gradient(x.get_item(i))
            
        return  BlockDataContainer(*out)     
        
    def __getitem__(self, ind):
        """Returns the function of the separable sum at the :code:`ind` place.
        """
        return self.functions[ind]
        
    def __rmul__(self, other):
        '''Define multiplication with a scalar
        
        :param other: number
        Returns a new `BlockFunction`_ containing the product of the scalar with all the functions in the block
        '''
        if not isinstance(other, Number):
            raise NotImplemented
        return BlockFunction( * [ other * el for el in self.functions] )

                            
    
    
