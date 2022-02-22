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


from numbers import Number
import numpy as np

class Function(object):
    
    """ Abstract class representing a function.

    Attributes
    ----------
    L : positive :obj:`float`, default = `None`
        The Lipschitz constant of the gradient of the function, when it is differentiable.

    """
    
    def __init__(self, L = None):

        # value for the Lipschitz constant
        self._L = L
        
    def __call__(self,x):
                
        """ Abstract method that returns the value of the function F at :code:`x`.
        
        .. math:: F(x)

        """        
        
        raise NotImplementedError

    def gradient(self, x, out=None):
        
        """ Abstract method that returns the value of the gradient of function F at :code:`x` .
        
        .. math:: F'(x)
                    
        """    
        raise NotImplementedError

    def proximal(self, x, tau, out=None):

        r""" Abstract method that returns the proximal operator of the function F at :code:`x` .
        
        .. math:: \mathrm{prox}_{\tau F}(x) = \underset{z}{\mathrm{argmin}}  \frac{1}{2\tau}\|z - x\|^{2} + F(z)
                    
        """            
       
        raise NotImplementedError

    def convex_conjugate(self, x):

        r""" Abstract method that returns the value of the convex conjugate of the function F at :code:`x` .
        
        .. math:: F^{*}(x^{*}) = \underset{x^{*}}{\sup} <x^{*}, x> - F(x)
                    
        """     

        raise NotImplementedError

    def proximal_conjugate(self, x, tau, out = None):

        r""" Abstract method that returns the proximal operator of the convex conjugate of the function F at :code:`x` .

        .. math:: \mathrm{prox}_{\tau F^{*}}(x^{*}) = \underset{z^{*}}{\mathrm{argmin}} \frac{1}{2}\|z^{*} - x^{*}\|^{2} + \tau F^{*}(z^{*})

        Moreau's decomposition is used to compute the proximal operator of the convex conjugate F^{*} using the proximal operator of F.
        
        .. math:: \mathrm{prox}_{\tau F^{*}}(x) = x - \tau\mathrm{prox}_{\tau^{-1} F}(\tau^{-1}x)
                         

        Parameters
        ----------
        x : DataContainer
            Point where `proximal_conjugate` is evaluated.
        tau : positive :obj:`float` 
            Step size       
        
        Returns
        -------

        DataContainer 
            Returns the proximal operator of the convex conjugate of the function F at :code:`x` .
   
        """            
        
        try:
            tmp = x
            x.divide(tau, out = tmp)
        except TypeError:
            tmp = x.divide(tau, dtype=np.float32)

        if out is None:
            val = self.proximal(tmp, 1.0/tau)
        else:            
            self.proximal(tmp, 1.0/tau, out = out)
            val = out
                   
        if id(tmp) == id(x):
            x.multiply(tau, out = x)

        # CIL issue #1078, cannot use axpby
        # val.axpby(-tau, 1.0, x, out=val)
        val.multiply(-tau, out = val)
        val.add(x, out = val)

        if out is None:
            return val

    # Algebra for the Function Class
    
        # Sum two functions
        # Subtract two functions
        # Add/Substract Function with Scalar
        # Multiply with Scalar
    
    def __add__(self, other):
        
        """ Returns the sum of the functions.
        
            *  Sum of two functions :math:`(F_{1}+F_{2})(x) = F_{1}(x) + F_{2}(x)`
            *  Sum of a function with a scalar c :math:`(F_{1} + c)(x) = F_{1}(x) + c`

        """
        
        if isinstance(other, Function):
            return SumFunction(self, other)
        elif isinstance(other, (SumScalarFunction, ConstantFunction, Number)):
            return SumScalarFunction(self, other)
        else:
            raise ValueError('Not implemented')   
            
    def __radd__(self, other):        
        """ Addition is commutative. """
        return self + other 
                          
    def __sub__(self, other):
        """ Subtract two functions."""
        return self + (-1) * other    

    def __rmul__(self, scalar):
        """Returns a function multiplied by a scalar."""               
        return ScaledFunction(self, scalar)
    
    def __mul__(self, scalar):
        """Multiply a function with a scalar."""  
        return self.__rmul__(scalar)
    
    def centered_at(self, center):
        """ Returns a function translated from the origin, at the :code:'center' point."""
        if center is None:
            return self
        else:
            return TranslateFunction(self, center)
    
    @property
    def L(self):
        """ Lipschitz constant of the gradient of function F.
        """
        return self._L

    @L.setter
    def L(self, value):
        """Setter for the Lipschitz constant"""
        if isinstance(value, (Number,)) and value >= 0:
            self._L = value
        else:
            raise TypeError('The Lipschitz constant is a real positive number')
    
class SumFunction(Function):
    
    """ Abstract class representing the sum of functions.

    Parameters
    ----------

    self : Function
           Left function
    other : Function
           Right function

    Returns
    -------
    SumFunction
        The sum of two functions.

    .. math:: (F_{1} + F_{2})(x)  = F_{1}(x) + F_{2}(x)
    
    """
    
    def __init__(self, function1, function2 ):
                
        super(SumFunction, self).__init__()        

        self.function1 = function1
        self.function2 = function2

    @property
    def L(self):
        """ Lipschitz constant 
        
        .. math:: L = L_{1} + L_{2}, L_{1}, L_{2} the corresponding Lipschitz contants.
        
        """
        if self.function1.L is not None and self.function2.L is not None:
            self._L = self.function1.L + self.function2.L
        else:
            self._L = None
        return self._L
    @L.setter
    def L(self, value):
        # call base class setter
        super(SumFunction, self.__class__).L.fset(self, value )

    def __call__(self,x):

        r"""Returns the value of the sum of functions :math:`F_{1}` and :math:`F_{2}` at x
        
        .. math:: (F_{1} + F_{2})(x) = F_{1}(x) + F_{2}(x)
                
        """  
        return self.function1(x) + self.function2(x)
    
    def gradient(self, x, out=None):
        
        r"""Returns the value of the sum of the gradient of functions :math:`F_{1}` and :math:`F_{2}` at x, if both of them are differentiable

        .. math:: (F'_{1} + F'_{2})(x)  = F'_{1}(x) + F'_{2}(x)
        
        """
        
        if out is None:            
            return self.function1.gradient(x) +  self.function2.gradient(x)  
        else:

            self.function1.gradient(x, out=out)
            out.add(self.function2.gradient(x), out=out)                
            
class ScaledFunction(Function):


    r""" Abstract class representing the scalar multiplication with a Function.

    Parameters
    ----------

    self : Function
           Function that is multiplied with a scalar
    scalar : :obj:`float`
           Scalar

    Returns
    -------
    ScaledFunction
        Scalar multiplication with a function.

    .. math:: G  = \alpha F
    
    """    
    
    def __init__(self, function, scalar):
        
        super(ScaledFunction, self).__init__() 
                                                     
        if not isinstance (scalar, Number):
            raise TypeError('Expected scalar. Got {}'.format(type(scalar)))
        
        self.scalar = scalar
        self.function = function  

    @property
    def L(self):
        if self._L is None:
            if self.function.L is not None:
                self._L = abs(self.scalar) * self.function.L
            else:
                self._L = None
        return self._L

    @L.setter
    def L(self, value):
        # call base class setter
        super(ScaledFunction, self.__class__).L.fset(self, value )

    @property
    def scalar(self):
        return self._scalar

    @scalar.setter
    def scalar(self, value):
        if isinstance(value, (Number, )):
            self._scalar = value
        else:
            raise TypeError('Expecting scalar. Got {}'.format(type(value)))

    def __call__(self,x, out=None):

        r"""Returns the value of the scaled function at :code:'x'.
        
        .. math:: G(x) = \alpha F(x)
        
        """
        return self.scalar * self.function(x)

    def convex_conjugate(self, x):

        r"""Returns the value of the convex conjugate of the scaled function at :code:'x'.
        
        .. math:: G^{*}(x) = \alpha  F^{*}(\frac{x}{\alpha})
        
        """

        try:
            x.divide(self.scalar, out = x)
            tmp = x
        except TypeError:
            tmp = x.divide(self.scalar, dtype=np.float32)

        val = self.function.convex_conjugate(tmp)

        if id(tmp) == id(x):
            x.multiply(self.scalar, out = x)

        return  self.scalar * val

    
    def gradient(self, x, out=None):

        r"""Returns the value of the gradient of the scaled function at :code:'x'.
        
        .. math:: G'(x) = \alpha  F'(x)
        
        """
        if out is None:            
            return self.scalar * self.function.gradient(x)
        else:
            self.function.gradient(x, out=out)
            out *= self.scalar  

    def proximal(self, x, tau, out=None):
        
        r"""Returns the proximal operator of the scaled function at :code:'x'.
        
        .. math:: \mathrm{prox}_{\tau G}(x) = \mathrm{prox}_{(\tau\alpha) F}(x)
        
        """        

        return self.function.proximal(x, tau*self.scalar, out=out)     


    def proximal_conjugate(self, x, tau, out = None):

        r"""Returns the proximal operator of the convex conjugate of the scaled function at :code:'x'
        using the Moreau Decomposition.
        """  

        try:
            tmp = x
            x.divide(tau, out = tmp)
        except TypeError:
            tmp = x.divide(tau, dtype=np.float32)

        if out is None:
            val = self.function.proximal(tmp, self.scalar/tau )
        else:
            self.function.proximal(tmp, self.scalar/tau, out = out)
            val = out     

        if id(tmp) == id(x):
            x.multiply(tau, out = x)

        # CIL issue #1078, cannot use axpby
        #val.axpby(-tau, 1.0, x, out=val)
        val.multiply(-tau, out = val)
        val.add(x, out = val)

        if out is None:
            return val

class SumScalarFunction(SumFunction):

    r""" Abstract class representing the sum of a Function with a scalar.

    Parameters
    ----------

    self : Function
           Function is added with a scalar
    constant : :obj:`float`
           Scalar

    Returns
    -------
    SumScalarFunction
        Sum of a Function with a scalar.

    .. math:: G  = F + constant
    
    """        
             
    def __init__(self, function, constant):
        
        super(SumScalarFunction, self).__init__(function, ConstantFunction(constant))        
        self.constant = constant
        self.function = function
        
    def convex_conjugate(self,x):

        r"""Returns the value of the convex conjugate of the sum of function with a scaled at :code:'x'.
        
        .. math:: (G)^{*}(x^{*}) = F^{*}(x^{*}) - scalar
        
        """        
                 
        return self.function.convex_conjugate(x) - self.constant
    
    def proximal(self, x, tau, out=None):

        r"""Returns the proximal operator of the scaled function at :code:'x'.
        
        .. math:: \mathrm{prox}_{\tau G}(x) = \mathrm{prox}_{\tau F}(x)
        
        """            
                       
        return self.function.proximal(x, tau, out=out)        
    
    @property
    def L(self):
        if self._L is None:
            if self.function.L is not None:
                self._L = self.function.L
            else:
                self._L = None
        return self._L
    @L.setter
    def L(self, value):
        # call base class setter
        super(SumScalarFunction, self.__class__).L.fset(self, value )

class ConstantFunction(Function):

    r"""  Function representing the Constant Function.

    Parameters
    ----------

    constant : :obj:`float`
           Contant value of the function.

    .. math:: F = constant
    
    """     
    
    def __init__(self, constant = 0):
        self.constant = constant
        super(ConstantFunction, self).__init__(L=0)
         
    def __call__(self,x):
        
        r""" Returns the value of the constant function at :code:'x'.
        
        .. math:: F(x) = constant
        
        """
        return self.constant
        
    def gradient(self, x, out=None):
        
        r""" Returns the value of the gradient of the constant function at :code:'x'.
        
         .. math:: F'(x) = 0
         
        """       
        if out is None:
            return x * 0.
        else:
            out.fill(0)
    
    def convex_conjugate(self, x):

        r""" Returns the value of the convex conjugate of the constant function at :code:'x'.
        
        .. math:: 
            F^{*}(x^{*}) 
            =
            \begin{cases}
                - constant, & \mbox{ if } x^{*} = 0\\
                \infty, & \mbox{otherwise}
            \end{cases}

        Note
        ----
        However, :math:`x^{*} = 0` is true only in the limit of iterations, so in fact this can be infinity.
        To avoid infinity values in the convex conjugate, we penalise this value accordingly.
        The following penalisation is useful when computing objectives, e.g., primal and dual objectives in the PDHG algorithm, for convergence purposes.
        
        .. math:: F^{*}(x^{*}) = \sum \max\{x^{*}, 0\}
                         
        """           
        return x.maximum(0).sum()
                
    def proximal(self, x, tau, out=None):
        
        """Returns the proximal operator of the constant function at :code:'x'.
        
        .. math:: \mathrm{prox}_{\tau F}(x) = x 
        
        """
        if out is None:
            return x.copy()
        else:
            out.fill(x)
        
    @property
    def constant(self):
        return self._constant
    @constant.setter
    def constant(self, value):
        if not isinstance (value, Number):
            raise TypeError('Expected scalar. Got {}'.format(type(value)))
        self._constant = value
    @property
    def L(self):
        return 0.
    def __rmul__(self, other):
        '''defines the right multiplication with a number'''
        if not isinstance (other, Number):
            raise NotImplemented
        constant = self.constant * other
        return ConstantFunction(constant)

class ZeroFunction(ConstantFunction):
    
    """ Function representing the Zero Function. 
    
    .. math:: F = 0

    """
    
    def __init__(self):
        super(ZeroFunction, self).__init__(constant = 0.) 
        
class TranslateFunction(Function):

    r""" Abstract class representing the translation of the Function.

    Parameters
    ----------

    self : Function
           Function to be translated
    center : :obj:`float`
           Point of translation

    Returns
    -------
    TranslateFunction
        Translates the Function from the center.

    .. math:: G  = F(\cdot - center) 
    
    """      
        
    def __init__(self, function, center):
        try:
            L = function.L
        except NotImplementedError as nie:
            L = None
        super(TranslateFunction, self).__init__(L = L) 
                        
        self.function = function
        self.center = center
        
    def __call__(self, x):
        
        r"""Returns the value of the translated function at :code:`x`.
        
        .. math:: G(x) = F(x - b)
        
        """        
        try:
            x.subtract(self.center, out = x)
            tmp = x
        except TypeError:
            tmp = x.subtract(self.center, dtype=np.float32)

        val = self.function(tmp)

        if id(tmp) == id(x):
            x.add(self.center, out = x)

        return val

    
    def gradient(self, x, out = None):
        
        r"""Returns the value of the gradient of the translated function :code:`x`.
        
        .. math:: G'(x) =  F'(x - b)
        
        """        
        try:
            x.subtract(self.center, out = x)
            tmp = x
        except TypeError:
            tmp = x.subtract(self.center, dtype=np.float32)

        if out is None:
            val = self.function.gradient(tmp)
        else:                       
            self.function.gradient(tmp, out = out)   

        if id(tmp) == id(x):
            x.add(self.center, out = x)

        if out is None:
            return val

    
    def proximal(self, x, tau, out = None):
        
        r"""Returns the proximal operator of the translated function at :code:`x`.
        
        .. math:: \mathrm{prox}_{\tau G}(x) = \mathrm{prox}_{\tau F}(x-b) + b
        
        """        
        try:
            x.subtract(self.center, out = x)
            tmp = x
        except TypeError:
            tmp = x.subtract(self.center, dtype=np.float32)

        if out is None:
            val = self.function.proximal(tmp, tau)
            val.add(self.center, out = val)
        else:                    
            self.function.proximal(tmp, tau, out = out)   
            out.add(self.center, out = out)

        if id(tmp) == id(x):
            x.add(self.center, out = x)

        if out is None:
            return val

    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the translated function at :code:`x`.
        
        .. math:: G^{*}(x^{*}) = F^{*}(x^{*}) + <x^{*}, b >
        
        """        
        
        return self.function.convex_conjugate(x) + self.center.dot(x)
