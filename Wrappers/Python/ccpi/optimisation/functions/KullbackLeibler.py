# -*- coding: utf-8 -*-
# Copyright 2019 Science Technology Facilities Council
# Copyright 2019 University of Manchester
#
# This work is part of the Core Imaging Library developed by Science Technology
# Facilities Council and University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
from ccpi.optimisation.functions import Function
from numbers import Number
import functools
import scipy.special
try:
    import numba
    from numba import jit, prange
    has_numba = True
    '''Some parallelisation of KL calls'''
    @jit(nopython=True)
    def kl_proximal(x,b, tau, out, eta):
        for i in prange(x.size):
            X = x.flat[i]
            E = eta.flat[i]
            out.flat[i] = 0.5 *  ( 
                ( X - E - tau ) +\
                numpy.sqrt( (X + E - tau)**2. + \
                    (4. * tau * b.flat[i]) 
                )
            )
    @jit(nopython=True)
    def kl_proximal_arr(x,b, tau, out, eta):
        for i in prange(x.size):
            t = tau.flat[i]
            X = x.flat[i]
            E = eta.flat[i]
            out.flat[i] = 0.5 *  ( 
                ( X - E - t ) +\
                numpy.sqrt( (X + E - t)**2. + \
                    (4. * t * b.flat[i]) 
                )
            )
        
    @jit(nopython=True)
    def kl_proximal_conjugate_arr(x, b, eta, tau, out):
        #z = x + tau * self.bnoise
        #return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
        for i in prange(x.size):
            t = tau.flat[i]
            z = x.flat[i] + ( t * eta.flat[i] )
            out.flat[i] = 0.5 * ( 
                (z + 1) - numpy.sqrt((z-1)*(z-1) + 4 * t * b.flat[i])
                )
        
    @jit(nopython=True)
    def kl_proximal_conjugate(x, b, eta, tau, out):
        #z = x + tau * self.bnoise
        #return 0.5*((z + 1) - ((z-1)**2 + 4 * tau * self.b).sqrt())
        for i in prange(x.size):
            z = x.flat[i] + ( tau * eta.flat[i] )
            out.flat[i] = 0.5 * ( 
                (z + 1) - numpy.sqrt((z-1)*(z-1) + 4 * tau * b.flat[i])
                )
    @jit(nopython=True)
    def kl_gradient(x, b, out, eta):
        for i in prange(x.size):
            out.flat[i] = 1 - b.flat[i]/(x.flat[i] + eta.flat[i])

    @jit(nopython=True)
    def kl_div(x, y, eta):
        accumulator = 0.
        for i in prange(x.size):
            X = x.flat[i]
            Y = y.flat[i] + eta.flat[i]
            if X > 0 and Y > 0:
                # out.flat[i] = X * numpy.log(X/Y) - X + Y
                accumulator += X * numpy.log(X/Y) - X + Y
            elif X == 0 and Y >= 0:
                # out.flat[i] = Y
                accumulator += Y
            else:
                # out.flat[i] = numpy.inf
                return numpy.inf
        return accumulator
    @jit(nopython=True)
    def kl_div_mask(x, y, eta, mask):
        accumulator = 0.
        for i in prange(x.size):
            if mask.flat[i] > 0:
                X = x.flat[i]
                Y = y.flat[i] + eta.flat[i]
                if X > 0 and Y > 0:
                    # out.flat[i] = X * numpy.log(X/Y) - X + Y
                    accumulator += X * numpy.log(X/Y) - X + Y
                elif X == 0 and Y >= 0:
                    # out.flat[i] = Y
                    accumulator += Y
                else:
                    # out.flat[i] = numpy.inf
                    return numpy.inf
        return accumulator

    @jit(nopython=True)
    def kl_convex_conjugate(x, b, eta):
        accumulator = 0.
        for i in prange(x.size):
            X = b.flat[i]
            x_f = x.flat[i]
            Y = 1 - x_f
            if Y > 0:
                if X > 0:
                    # out.flat[i] = X * numpy.log(X/Y) - X + Y
                    accumulator += X * numpy.log(Y)
                # else xlogy is 0 so it doesn't add to the accumulator
                accumulator += eta.flat[i] * x_f
        return - accumulator
    
    # force a jit
    print ("forcing jit in KL")
    x = numpy.asarray(numpy.random.random((10,10)), dtype=numpy.float32)
    b = numpy.asarray(numpy.random.random((10,10)), dtype=numpy.float32)
    eta = numpy.zeros_like(x)
    out = numpy.empty_like(x)
    mask = x > 0.3
    tau = 1.
    tauarr = numpy.ones_like(x)
    kl_div(b, x, eta)
    kl_div_mask(b, x, eta, mask)
    kl_gradient(x, b, out, eta)
    kl_proximal(x, b, tau, out, eta)
    kl_proximal_arr(x, b, tauarr, out, eta)
    kl_proximal_conjugate(x, b, eta, tau, out)
    kl_proximal_conjugate_arr(x, b, eta, tauarr, out)
    kl_convex_conjugate(x, b, eta)
    
except ImportError as ie:
    has_numba = False

class KullbackLeibler(Function):
    
    r""" Kullback Leibler divergence function is defined as:
            
    .. math:: F(u, v)
            = \begin{cases} 
            u \log(\frac{u}{v}) - u + v & \mbox{ if } u > 0, v > 0\\
            v & \mbox{ if } u = 0, v \ge 0 \\
            \infty, & \mbox{otherwise}
            \end{cases}  
            
    where we use the :math:`0\log0 := 0` convention. 

    At the moment, we use build-in implemention of scipy, see
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.kl_div.html
    
    The Kullback-Leibler function is used as a fidelity term in minimisation problems where the
    acquired data follow Poisson distribution. If we denote the acquired data with :math:`b`
    then, we write
    
     .. math:: \underset{i}{\sum} F(b_{i}, (v + \eta)_{i})
     
     where, :math:`\eta` is an additional noise. 
     
     Example: In the case of Positron Emission Tomography reconstruction :math:`\eta` represents 
     scatter and random events contribution during the PET acquisition. Hence, in that case the KullbackLeibler
     fidelity measures the distance between :math:`\mathcal{A}v + \eta` and acquisition data :math:`b`, where
     :math:`\mathcal{A}` is the projection operator.
     
     This is related to PoissonLogLikelihoodWithLinearModelForMean definition that is used in PET reconstruction
     in the PET-MR software , see https://github.com/CCPPETMR and for more details in
    
    http://stir.sourceforge.net/documentation/doxy/html/classstir_1_1PoissonLogLikelihoodWithLinearModelForMean.html
                        
    """            
    
    
    def __init__(self,  **kwargs):
        
        super(KullbackLeibler, self).__init__(L = None)          
        self.b = kwargs.get('b', None)
        
        if self.b is None:
            raise ValueError('Please define data, as b = ...')
            
        if (self.b.as_array() < 0).any():            
            raise ValueError('Data should be larger or equal to 0')              
         
        self.eta = kwargs.get('eta',self.b * 0.0)
        self.use_numba = kwargs.get('use_numba', True)
        if self.use_numba and has_numba:
            self.b_np = self.b.as_array()
            self.eta_np = self.eta.as_array()
        mask = kwargs.get('mask', None)
        if hasattr (mask, 'as_array'):
            self.mask = mask.as_array()
        else:
            self.mask = mask

        if self.mask is not None and not ( self.use_numba and has_numba) :
            print ('Cannot make use of mask without numba')
        
                                                    
    def __call__(self, x):
        

        r"""Returns the value of the KullbackLeibler function at :math:`(b, x + \eta)`.
        To avoid infinity values, we consider only pixels/voxels for :math:`x+\eta\geq0`.
        """
        if self.use_numba and has_numba:
            # tmp = numpy.empty_like(x.as_array())
            if self.mask is not None:
                return kl_div_mask(self.b_np, x.as_array(), self.eta_np, self.mask)
            return kl_div(self.b_np, x.as_array(), self.eta_np)
        else: 
            tmp_sum = (x + self.eta).as_array()
            ind = tmp_sum >= 0
            tmp = scipy.special.kl_div(self.b.as_array()[ind], tmp_sum[ind])             
            return numpy.sum(tmp)         
        
    def log(self, datacontainer):
        '''calculates the in-place log of the datacontainer'''
        if not functools.reduce(lambda x,y: x and y>0, datacontainer.as_array().ravel(), True):
            raise ValueError('KullbackLeibler. Cannot calculate log of negative number')
        datacontainer.fill( numpy.log(datacontainer.as_array()) )

        
    def gradient(self, x, out=None):
        
        r"""Returns the value of the gradient of the KullbackLeibler function at :math:`(b, x + \eta)`.                
        
        .. math:: F'(b, x + \eta) = 1 - \frac{b}{x+\eta}
        
        We require the :math:`x+\eta>0` otherwise we have inf values.
        
        """     
        if self.use_numba and has_numba:
            if out is None:
                out = (x * 0.)
                out_np = out.as_array()
                kl_gradient(x.as_array(), self.b.as_array(), out_np, self.eta.as_array())
                # out.fill(out_np)
                return out
            else:
                out_np = out.as_array()
                kl_gradient(x.as_array(), self.b.as_array(), out_np, self.eta.as_array())
                # out.fill(out_np)
        else:                           
            tmp_sum_array = (x + self.eta).as_array()
            if out is None:   
                tmp_out = x.geometry.allocate() 
                tmp_out.as_array()[tmp_sum_array>0] = 1 - self.b.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]         
                return tmp_out        
            else:                 
                x.add(self.eta, out=out)
                out.as_array()[tmp_sum_array>0] = 1 - self.b.as_array()[tmp_sum_array>0]/tmp_sum_array[tmp_sum_array>0]                                

            
    def convex_conjugate(self, x):
        
        r"""Returns the value of the convex conjugate of the KullbackLeibler function at :math:`(b, x + \eta)`.                
        
        .. math:: F^{*}(b, x + \eta) = - b \log(1-x^{*}) - <x^{*}, \eta> 
        
        """  
        if self.use_numba and has_numba:
            return kl_convex_conjugate(x.as_array(), self.b_np, self.eta_np)
        else:
            tmp = 1 - x.as_array()
            ind = tmp>0
            xlogy = - scipy.special.xlogy(self.b.as_array()[ind], tmp[ind])  
            return numpy.sum(xlogy) - self.eta.dot(x)
            
    def proximal(self, x, tau, out=None):
        
        r"""Returns the value of the proximal operator of the KullbackLeibler function at :math:`(b, x + \eta)`.
        
        .. math:: \mathrm{prox}_{\tau F}(x) = \frac{1}{2}\bigg( (x - \eta - \tau) + \sqrt{ (x + \eta - \tau)^2 + 4\tau b} \bigg)
        
        The proximal for the convex conjugate of :math:`F` is 
        
        .. math:: \mathrm{prox}_{\tau F^{*}}(x) = 0.5*((z + 1) - \sqrt{(z-1)^2 + 4 * \tau b})
        
        where :math:`z = x + \tau \eta`
                    
        """
        if self.use_numba and has_numba:
            if out is None:
                out = (x * 0.)
                # out_np = numpy.empty_like(out.as_array(), dtype=numpy.float64)
                out_np = out.as_array()
                if isinstance(tau, Number):
                    kl_proximal(x.as_array(), self.b_np, tau, out_np, self.eta_np)
                else:
                    # it should be a DataContainer
                    kl_proximal_arr(x.as_array(), self.b_np, tau.as_array(), out_np, self.eta_np)
                out.fill(out_np)
                return out
            else:
                out_np = out.as_array()
                if isinstance(tau, Number):
                    kl_proximal(x.as_array(), self.b_np, tau, out_np, self.eta_np)
                else:
                    # it should be a DataContainer
                    kl_proximal_arr(x.as_array(), self.b_np, tau.as_array(), out_np, self.eta_np)
                out.fill(out_np)                    
        else:
            if out is None:        
                return 0.5 *( (x - self.eta - tau) + ( (x + self.eta - tau).power(2) + 4*tau*self.b   ) .sqrt() )        
            else:                      
                x.add(self.eta, out=out)
                out -= tau
                out *= out
                out.add(self.b * (4 * tau), out=out)
                out.sqrt(out=out)  
                out.subtract(tau, out=out)
                out.subtract(self.eta, out=out)
                out.add(x, out=out)         
                out *= 0.5            
        
                            
    def proximal_conjugate(self, x, tau, out=None):
        
        r'''Proximal operator of the convex conjugate of KullbackLeibler at x:
           
           .. math::     prox_{\tau * f^{*}}(x)
        '''

        if self.use_numba and has_numba:
            if out is None:
                out = (x * 0.)
                # out_np = numpy.empty(out.shape, dtype=numpy.float64)
                out_np = out.as_array()
                if isinstance(tau, Number):
                    kl_proximal_conjugate(x.as_array(), self.b_np, self.eta_np, tau, out_np)
                else:
                    kl_proximal_conjugate_arr(x.as_array(), self.b_np, self.eta_np, tau.as_array(), out_np)
                out.fill(out_np)
                return out
            else:
                out_np = out.as_array()
                if isinstance(tau, Number):
                    kl_proximal_conjugate(x.as_array(), self.b_np, self.eta_np, tau, out_np)
                else:
                    kl_proximal_conjugate_arr(x.as_array(), self.b_np, self.eta_np, tau.as_array(), out_np)
                out.fill(out_np)                    
        else:
            if out is None:
                z = x + tau * self.eta
                return 0.5*((z + 1) - ((z-1).power(2) + 4 * tau * self.b).sqrt())
            else:            
                tmp = tau * self.eta
                tmp += x
                tmp -= 1
                
                self.b.multiply(4*tau, out=out)    
                
                out.add(tmp.power(2), out=out)
                out.sqrt(out=out)
                out *= -1
                tmp += 2
                out += tmp
                out *= 0.5




if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry
    import numpy as np
    
    M, N, K =  30, 30, 20
    ig = ImageGeometry(N, M, K)
    
    u1 = ig.allocate('random', seed = 500)    
    g1 = ig.allocate('random', seed = 100)
    b1 = ig.allocate('random', seed = 500)
    
    # with no data
    try:
        f = KullbackLeibler()   
    except ValueError:
        print('Give data b=...\n')
        
    print('With negative data, no background\n')   
    try:        
        f = KullbackLeibler(b=-1*g1)
    except ValueError:
        print('We have negative data\n') 
        
    f = KullbackLeibler(b=g1)        
        
    print('Check KullbackLeibler(x,x)=0\n') 
    numpy.testing.assert_equal(0.0, f(g1))
            
    print('Check gradient .... is OK \n')
    res_gradient = f.gradient(u1)
    res_gradient_out = u1.geometry.allocate()
    f.gradient(u1, out = res_gradient_out) 
    numpy.testing.assert_array_almost_equal(res_gradient.as_array(), \
                                            res_gradient_out.as_array())  
    
    print('Check proximal ... is OK\n')        
    tau = 0.4
    res_proximal = f.proximal(u1, tau)
    res_proximal_out = u1.geometry.allocate()   
    f.proximal(u1, tau, out = res_proximal_out)
    numpy.testing.assert_array_almost_equal(res_proximal.as_array(), \
                                            res_proximal_out.as_array())  
                

    print('Check KullbackLeibler with background\n')       
    f1 = KullbackLeibler(b=g1, eta=b1) 
        
    tmp_sum = (u1 + f1.eta).as_array()
    ind = tmp_sum >= 0
    tmp = scipy.special.kl_div(f1.b.as_array()[ind], tmp_sum[ind])                 
    numpy.testing.assert_almost_equal(f1(u1), numpy.sum(tmp), decimal=1)
    
    print('Check proximal KL without background\n')   
    tau = [0.1, 1, 10, 100, 10000]
    
    for t1 in tau:
        
        proxc = f.proximal_conjugate(u1,t1)
        proxc_out = ig.allocate()
        f.proximal_conjugate(u1, t1, out = proxc_out)
        print('tau = {} is OK'.format(t1) )
        numpy.testing.assert_array_almost_equal(proxc.as_array(), 
                                                proxc_out.as_array(),
                                                decimal = 4)
        
    print('\nCheck proximal KL with background\n')          
    for t1 in tau:
        
        proxc1 = f1.proximal_conjugate(u1,t1)
        proxc_out1 = ig.allocate()
        f1.proximal_conjugate(u1, t1, out = proxc_out1)
        numpy.testing.assert_array_almost_equal(proxc1.as_array(), 
                                                proxc_out1.as_array(),
                                                decimal = 4)  
    
        print('tau = {} is OK'.format(t1) )    
        
    f = KullbackLeibler(b=g1, eta = b1)        
    tau = 0.4
    res_proximal = f.proximal(u1, tau)
    res_proximal_out = u1.geometry.allocate()   
    f.proximal(u1, tau, out = res_proximal_out)
    numpy.testing.assert_array_almost_equal(res_proximal.as_array(), \
                                            res_proximal_out.as_array())  
    print('Check proximal with eta ... is OK\n')         
        
    