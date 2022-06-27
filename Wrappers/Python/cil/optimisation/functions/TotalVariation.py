# Copyright 2022 United Kingdom Research and Innovation
# Copyright 2022 The University of Manchester

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from cil.optimisation.functions import Function, IndicatorBox, MixedL21Norm
from cil.optimisation.operators import GradientOperator
import numpy as np
from numbers import Number
import warnings



class TotalVariation(Function):
    
    r""" Total variation Function

    .. math:: \mathrm{TV}(u) := \|\nabla u\|_{2,1} = \sum \|\nabla u\|_{2},\, (\mbox{isotropic})

    .. math:: \mathrm{TV}(u) := \|\nabla u\|_{1,1} = \sum \|\nabla u\|_{1}\, (\mbox{anisotropic})

    Notes
    -----

    The :code:`TotalVariation` (TV) :code:`Function` acts as a compositite function, i.e.,
    the composition of the :class:`.MixedL21Norm` fuction and the :class:`.GradientOperator` operator,

    .. math:: f(u) = \|u\|_{2,1}, \Rightarrow (f\circ\nabla)(u) = f(\nabla x) = \mathrm{TV}(u)

    In that case, the proximal operator of TV does not have an exact solution and we use an iterative 
    algorithm to solve:

    .. math:: \mathrm{prox}_{\tau \mathrm{TV}}(b) := \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TV}(u)
    
    The algorithm used for the proximal operator of TV is the Fast Gradient Projection algorithm (or FISTA)
    applied to the _dual problem_ of the above problem, see :cite:`BeckTeboulle_b`, :cite:`BeckTeboulle_a`.




    Parameters
    ----------

    max_iteration : :obj:`int`, default = 100
        Maximum number of iterations for the FGP algorithm.
    tolerance : :obj:`float`, default = None
        Stopping criterion for the FGP algorithm.
        
        .. math:: \|x^{k+1} - x^{k}\|_{2} < \mathrm{tolerance}

    correlation : :obj:`str`, default = `Space`
        Correlation between `Space` and/or `SpaceChannels` for the :class:`.GradientOperator`.
    backend :  :obj:`str`, default = `c`      
        Backend to compute the :class:`.GradientOperator`
    lower : :obj:`'float`, default = `-np.inf`
        A constraint is enforced using the :class:`.IndicatorBox` function, e.g., :code:`IndicatorBox(lower, upper)`.
    upper : :obj:`'float`, default = `np.inf`
        A constraint is enforced using the :class:`.IndicatorBox` function, e.g., :code:`IndicatorBox(lower, upper)`.  
    isotropic : :obj:`boolean`, default = True
        Use either isotropic or anisotropic definition of TV.

        .. math:: |x|_{2} = \sqrt{x_{1}^{2} + x_{2}^{2}},\, (\mbox{isotropic})

        .. math:: |x|_{1} = |x_{1}| + |x_{2}|\, (\mbox{anisotropic})

    split : :obj:`boolean`, default = False
        Splits the Gradient into spatial gradient and spectral or temporal gradient for multichannel data.

    info : :obj:`boolean`, default = False
        Information is printed for the stopping criterion of the FGP algorithm

    strong_convexity_constant : :obj:`float`, default = 0
        A strongly convex term weighted by the :code:`strong_convexity_constant` (:math:`\gamma`) parameter is added to the Total variation. 
        Now the :code:`TotalVariation` function is :math:`\gamma` - strongly convex and the proximal operator is

        .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2\tau}\|u - b\|^{2} + \mathrm{TV}(u) + \frac{\gamma}{2}\|u\|^{2} \Leftrightarrow

        .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2\frac{\tau}{1+\gamma\tau}}\|u - \frac{b}{1+\gamma\tau}\|^{2} + \mathrm{TV}(u) 

    Note
    ----

    In the case where the Total variation becomes a :math:`\gamma` - strongly convex function, i.e.,

    .. math:: \mathrm{TV}(u) + \frac{\gamma}{2}\|u\|^{2}

    :math:`\gamma` should be relatively small, so as the second term above will not act as an additional regulariser.
    For more information, see :cite:`Rasch2020`, :cite:`CP2011`.




    Examples
    --------

    .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + \alpha\|\nabla u\|_{2,1}

    >>> alpha = 2.0
    >>> TV = TotalVariation()
    >>> sol = TV.proxima(b, tau = alpha)

    Examples
    --------

    .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + \alpha\|\nabla u\|_{1,1} + \mathbb{I}_{C}(u)

    where :math:`C = \{1.0\leq u\leq 2.0\}`.

    >>> alpha = 2.0
    >>> TV = TotalVariation(isotropic=False, lower=1.0, upper=2.0)
    >>> sol = TV.proxima(b, tau = alpha)    


    Examples
    --------

    .. math:: \underset{u}{\mathrm{argmin}} \frac{1}{2}\|u - b\|^{2} + (\alpha\|\nabla u\|_{2,1} + \frac{\gamma}{2}\|u\|^{2})

    >>> alpha = 2.0
    >>> gamma = 1e-3
    >>> TV = alpha * TotalVariation(isotropic=False, strong_convexity_constant=gamma)
    >>> sol = TV.proxima(b, tau = 1.0)    



    """   
    
    
    def __init__(self,
                 max_iteration=100, 
                 tolerance = None, 
                 correlation = "Space",
                 backend = "c",
                 lower = -np.inf, 
                 upper = np.inf,
                 isotropic = True,
                 split = False,
                 info = False, 
                 strong_convexity_constant = 0):
        

        super(TotalVariation, self).__init__(L = None)
        # Regularising parameter = alpha
        self.regularisation_parameter = 1.
        
        # Iterations for FGP_TV
        self.iterations = max_iteration
        
        # Tolerance for FGP_TV
        self.tolerance = tolerance
        
        # Total variation correlation (isotropic=Default)
        self.isotropic = isotropic
        
        # correlation space or spacechannels
        self.correlation = correlation
        self.backend = backend        
        
        # Define orthogonal projection onto the convex set C
        self.lower = lower
        self.upper = upper
        self.projection_C = IndicatorBox(lower, upper).proximal
             
        # Setup GradientOperator as None. This is to avoid domain argument in the __init__     
        self._gradient = None
        self._domain = None

        self.pptmp = None
        self.pptmp1 = None
        
        # Print stopping information (iterations and tolerance error) of FGP_TV  
        self.info = info

        # splitting Gradient
        self.split = split

        # Strong convexity for TV
        self.strong_convexity_constant = strong_convexity_constant

    @property
    def regularisation_parameter(self):
        return self._regularisation_parameter

    @regularisation_parameter.setter
    def regularisation_parameter(self, value):
        if not isinstance(value, Number):
            raise TypeError("regularisation_parameter: expectec a number, got {}".format(type(value)))
        self._regularisation_parameter = value

    def __call__(self, x):
        
        r""" Returns the value of the TotalVariation function at :code:`x` ."""

        try:
            self._domain = x.geometry
        except:
            self._domain = x

        # Compute Lipschitz constant provided that domain is not None.
        # Lipschitz constant dependes on the GradientOperator, which is configured only if domain is not None
        if self._L is None:
            self.calculate_Lipschitz()

        if self.strong_convexity_constant>0:
            tmp = (self.strong_convexity_constant/2)*x.squared_norm()
        else:
            tmp = 0

        if self.isotropic:
            return self.regularisation_parameter * self.gradient.direct(x).pnorm(2).sum() + tmp
        else:
            return self.regularisation_parameter * self.gradient.direct(x).pnorm(1).sum() + tmp
    
                        
    def projection_P(self, x, out=None):
                       
        r""" Returns the proximal operator of the convex conjugate of the :class:`.MixedL21Norm` at :code:`x`.
        """
        
        # preallocated in proximal
        tmp = self.pptmp
        tmp1 = self.pptmp1
        tmp1 *= 0
        
        if self.isotropic:
            for el in x.containers:
                el.conjugate().multiply(el, out=tmp)
                tmp1.add(tmp, out=tmp1)
            tmp1.sqrt(out=tmp1)
            tmp1.maximum(1.0, out=tmp1)
            if out is None:
                return x.divide(tmp1)
            else:
                x.divide(tmp1, out=out)
        else:
            tmp1 = x.abs()
            tmp1.maximum(1.0, out=tmp1) 
            if out is None:
                return x.divide(tmp1)
            else:
                x.divide(tmp1, out=out)                   
    
    
    def proximal(self, x, tau, out = None):
        
        r""" Returns the proximal operator of the TotalVariation function at :code:`x` ."""        
        try:
            self._domain = x.geometry
        except:
            self._domain = x

        # Compute Lipschitz constant provided that domain is not None.
        # Lipschitz constant dependes on the GradientOperator, which is configured only if domain is not None
        if self._L is None:
            self.calculate_Lipschitz()            
        
        # initialise
        t = 1        
        tmp_p = self.gradient.range_geometry().allocate(0)  
        tmp_q = tmp_p.copy()
        tmp_x = self.gradient.domain_geometry().allocate(0)     
        p1 = self.gradient.range_geometry().allocate(0)

        if self.strong_convexity_constant>0:

            x /= (1 + tau*self.strong_convexity_constant)
            tau /= 1 + tau*self.strong_convexity_constant
            
        should_break = False
        for k in range(self.iterations):
                                                                                   
            t0 = t
            self.gradient.adjoint(tmp_q, out = tmp_x)
            
            # axpby now works for matrices
            tmp_x.axpby(-self.regularisation_parameter*tau, 1.0, x, out=tmp_x)
            self.projection_C(tmp_x, tau=None, out = tmp_x)                       

            self.gradient.direct(tmp_x, out=p1)
            if isinstance (tau, (Number, np.float32, np.float64)):
                p1 *= self.L/(self.regularisation_parameter * tau)
            else:
                p1 *= self.L/self.regularisation_parameter
                p1 /= tau

            if self.tolerance is not None:
                
                if k%5==0:
                    error = p1.norm()
                    p1 += tmp_q
                    error /= p1.norm()
                    if error<=self.tolerance:                           
                        should_break = True
                else:
                    p1 += tmp_q
            else:
                p1 += tmp_q
            if k == 0:
                # preallocate for projection_P
                self.pptmp = p1.get_item(0) * 0
                self.pptmp1 = self.pptmp.copy()

            self.projection_P(p1, out=p1)
            

            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
            
            #tmp_q.fill(p1 + (t0 - 1) / t * (p1 - tmp_p))
            p1.subtract(tmp_p, out=tmp_q)
            tmp_q *= (t0-1)/t
            tmp_q += p1
            
            tmp_p.fill(p1)

            if should_break:
                break
        
        # restore the values of inputs: tau , x
        if self.strong_convexity_constant>0:
            tau *= (1 + tau*self.strong_convexity_constant)
            x *= (1 + tau*self.strong_convexity_constant)

        #clear preallocated projection_P arrays
        self.pptmp = None
        self.pptmp1 = None
        
        # Print stopping information (iterations and tolerance error) of FGP_TV     
        if self.info:
            if self.tolerance is not None:
                print("Stop at {} iterations with tolerance {} .".format(k, error))
            else:
                print("Stop at {} iterations.".format(k))                
                    
        self.gradient.adjoint(tmp_q, out=tmp_x)
        tmp_x *= tau
        tmp_x *= self.regularisation_parameter 
        x.subtract(tmp_x, out=tmp_x)

        return self.projection_C(tmp_x, tau=None, out=out)  
    
    def convex_conjugate(self,x):   
        r""" Returns the value of convex conjugate of the TotalVariation function at :code:`x` ."""             
        return 0.0    
    
    def calculate_Lipschitz(self):
        r""" Default value for the Lipschitz constant."""
        
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        self._L = (1./self.gradient.norm())**2  
    
    @property
    def gradient(self):
        r""" GradientOperator is created if it is not instantiated yet. The domain of the `_gradient`,
        is created in the `__call__` and `proximal` methods. 

        """
        if self._domain is not None:
            self._gradient = GradientOperator(self._domain, correlation = self.correlation, backend = self.backend)
        else:
            raise ValueError(" The domain of the TotalVariation is {}. Please use the __call__ or proximal methods first before calling gradient.".format(self._domain))            

        return self._gradient

    def __rmul__(self, scalar):
        if not isinstance (scalar, Number):
            raise TypeError("scalar: Expectec a number, got {}".format(type(scalar)))
        self.regularisation_parameter = scalar
        return self
