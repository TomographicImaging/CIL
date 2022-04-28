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

from cil.optimisation.functions import Function, IndicatorBox
from cil.optimisation.operators import GradientOperator
import numpy as np
from numbers import Number
import warnings



class TotalVariation(Function):
    
    r'''Fast Gradient Projection algorithm for Total Variation(TV) regularisation
    
    .. math::  \min_{x} \frac{1}{2}||x-b||^{2}_{2} + \tau\alpha TV(x)
                
    Parameters:
      
      :param max_iteration: max iterations of FGP algorithm
      :type max_iteration: int, default 100
      :param tolerance: Stopping criterion
      :type tolerance: float, default `None` 
      :param correlation: Correlation between `Space` and/or `SpaceChannels` for the GradientOperator
      :type correlation: str, default 'Space'
      :param backend: Backend to compute finite differences for the GradientOperator
      :type backend: str, default 'c'
      :param lower: lower bound for the orthogonal projection onto the convex set C
      :type lower: Number, default `-np.inf`
      :param upper: upper bound for the orthogonal projection onto the convex set C
      :type upper: Number, default `+np.inf`
      :param isotropic: L2 norm is used for Gradient Operator (isotropic) 
      :type isotropic: bool, default `True` 
      
                        .. math:: \sum \sqrt{(\partial_y u)^{2} + (\partial_x u)^2} \mbox{ (isotropic) }
                        .. math:: \sum |\partial_y u| + |\partial_x u| \mbox{ (anisotropic) }
       
      :param split: splits the Gradient into spatial Gradient and spectral Gradient for multichannel data
      :type split: bool, default `False`           
      :param info: force a print to screen stating the stop
      :type info: bool, default `False`
      :Example:
 
      TV = alpha * TotalVariation()
      sol = TV.proximal(data, tau = 1.0) 
      .. note:: `tau` can be a number or an array. The latter case implies that step-size preconditioning is applied.
    Reference:
      
        A. Beck and M. Teboulle, "Fast Gradient-Based Algorithms for Constrained Total Variation 
        Image Denoising and Deblurring Problems," in IEEE Transactions on Image Processing,
        vol. 18, no. 11, pp. 2419-2434, Nov. 2009, 
        doi: 10.1109/TIP.2009.2028250.
        
    '''    
    
    
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
                 warmstart = False):
        

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
        self.tmp_proj_C = IndicatorBox(lower, upper).proximal
                        
        # Setup GradientOperator as None. This is to avoid domain argument in the __init__     

        self._gradient = None
        self._domain = None

        self.pptmp = None
        self.pptmp1 = None
        
        # Print stopping information (iterations and tolerance error) of FGP_TV  
        self.info = info

        # splitting Gradient
        self.split = split

        # warm-start
        self.warmstart  = warmstart
        if self.warmstart:
            self.hasstarted = False

    @property
    def regularisation_parameter(self):
        return self._regularisation_parameter

    @regularisation_parameter.setter
    def regularisation_parameter(self, value):
        if not isinstance(value, Number):
            raise TypeError("regularisation_parameter: expectec a number, got {}".format(type(value)))
        self._regularisation_parameter = value

    def __call__(self, x):
        
        r''' Returns the value of the \alpha * TV(x)'''
        try:
            self._domain = x.geometry
        except:
            self._domain = x
        # evaluate objective function of TV gradient
        if self.isotropic:
            return self.regularisation_parameter * self.gradient.direct(x).pnorm(2).sum()
        else:
            return self.regularisation_parameter * self.gradient.direct(x).pnorm(1).sum() 
    
    
    def projection_C(self, x, out=None):   
                     
        r''' Returns orthogonal projection onto the convex set C'''

        try:
            self._domain = x.geometry
        except:
            self._domain = x
        return self.tmp_proj_C(x, tau = None, out = out)
                        
    def projection_P(self, x, out=None):
                       
        r''' Returns the projection P onto \|\cdot\|_{\infty} '''  
        try:
            self._domain = x.geometry
        except:
            self._domain = x
        
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
        
        ''' Returns the solution of the FGP_TV algorithm '''         
        try:
            self._domain = x.geometry
        except:
            self._domain = x
        
        # initialise
        t = 1
        if not self.warmstart:      
            self.p1 = self.gradient.range_geometry().allocate(0)
        else:
            if not self.hasstarted:  
                self.p1 = self.gradient.range_geometry().allocate(0)
                self.hasstarted = True
        tmp_p = self.p1.copy() 
        tmp_q = self.p1.copy()
        tmp_x = self.gradient.domain_geometry().allocate(0)

        should_break = False
        for k in range(self.iterations):
                                                                                   
            t0 = t
            self.gradient.adjoint(tmp_q, out = tmp_x)
            
            # axpby now works for matrices
            tmp_x.axpby(-self.regularisation_parameter*tau, 1.0, x, out=tmp_x)
            self.projection_C(tmp_x, out = tmp_x)                       

            self.gradient.direct(tmp_x, out=self.p1)
            if isinstance (tau, (Number, np.float32, np.float64)):
                self.p1 *= self.L/(self.regularisation_parameter * tau)
            else:
                self.p1 *= self.L/self.regularisation_parameter
                self.p1 /= tau

            if self.tolerance is not None:
                
                if k%5==0:
                    error = self.p1.norm()
                    self.p1 += tmp_q
                    error /= self.p1.norm()
                    if error<=self.tolerance:                           
                        should_break = True
                else:
                    self.p1 += tmp_q
            else:
                self.p1 += tmp_q
            if k == 0:
                # preallocate for projection_P
                self.pptmp = self.p1.get_item(0) * 0
                self.pptmp1 = self.pptmp.copy()

            self.projection_P(self.p1, out=self.p1)
            

            t = (1 + np.sqrt(1 + 4 * t0 ** 2)) / 2
            
            #tmp_q.fill(self.p1 + (t0 - 1) / t * (self.p1 - tmp_p))
            self.p1.subtract(tmp_p, out=tmp_q)
            tmp_q *= (t0-1)/t
            tmp_q += self.p1
            
            tmp_p.fill(self.p1)

            if should_break:
                break
        
        #clear preallocated projection_P arrays
        self.pptmp = None
        self.pptmp1 = None
        
        # Print stopping information (iterations and tolerance error) of FGP_TV     
        if self.info:
            if self.tolerance is not None:
                print("Stop at {} iterations with tolerance {} .".format(k, error))
            else:
                print("Stop at {} iterations.".format(k))                
            
        if out is None:                        
            self.gradient.adjoint(tmp_q, out=tmp_x)
            tmp_x *= tau
            tmp_x *= self.regularisation_parameter 
            x.subtract(tmp_x, out=tmp_x)
            return self.projection_C(tmp_x)
        else:          
            self.gradient.adjoint(tmp_q, out = out)
            out*=tau
            out*=self.regularisation_parameter
            x.subtract(out, out=out)
            self.projection_C(out, out=out)
    
    def convex_conjugate(self,x):        
        return 0.0    
    @property
    def L(self):
        if self._L is None:
            self.calculate_Lipschitz()
        return self._L
    @L.setter
    def L(self, value):
        warnings.warn("You should set the Lipschitz constant with calculate_Lipschitz().")
        if isinstance(value, (Number,)) and value >= 0:
            self._L = value
        else:
            raise TypeError('The Lipschitz constant is a real positive number')

    def calculate_Lipschitz(self):
        # Compute the Lipschitz parameter from the operator if possible
        # Leave it initialised to None otherwise
        self._L = (1./self.gradient.norm())**2  
    
    @property
    def gradient(self):
        '''creates a gradient operator if not instantiated yet
        There is no check that the variable _domain is changed after instantiation (should not be the case)'''
        if self._gradient is None:
            if self._domain is not None:
                self._gradient = GradientOperator(self._domain, correlation = self.correlation, backend = self.backend)
        return self._gradient
    def __rmul__(self, scalar):
        if not isinstance (scalar, Number):
            raise TypeError("scalar: Expectec a number, got {}".format(type(scalar)))
        self.regularisation_parameter = scalar
        return self