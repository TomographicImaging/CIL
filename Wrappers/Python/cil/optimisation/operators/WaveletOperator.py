# -*- coding: utf-8 -*-
#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt

import numpy as np
import pywt # PyWavelets module
import warnings

from cil.optimisation.operators import LinearOperator
from cil.framework import VectorGeometry


###############################################################################
###############################################################################
########################## Discrete Wavelet Transform #########################
###############################################################################
###############################################################################

class WaveletOperator(LinearOperator):
    
    r'''                  
        Computes forward or inverse (adjoint) discrete wavelet transform of the input
        
        Parameters
        ----------
        :param domain_geometry: Domain geometry for the WaveletOperator

        [OPTIONAL PARAMETERS]
        :param range_geometry: Output geometry for the WaveletOperator. 
            Default = domain_geometry with the right coefficient array size deduced from pywt
        :param level: integer for decomposition level.
            Default = log_2(min(shape(axes))), i.e. the maximum number of accurate downsamplings possible
        :type wname: string label for wavelet used.
            Default = "haar"
        :type axes: range of ints to define the dimensions to decompose along. Note that channel is the first dimension:
            For example, spatial DWT is given by axes=range(1,3) and channelwise DWT is axes=range(1)
            Default = None, meaning all dimensions are transformed. Same as axes = range(ndim)
        **kwargs:
            correlation: str, default 'All'. Only applied if 'axes' = None!
                'All' will compute the wavelet decomposition on every possible dimension.
                'Space' will compute the wavelet decomposition on only the spatial dimensions. If there are multiple channels, each channel is decomposed independently.
                'Channels' will compute the wavelet decomposition on only the channels, independently for every spatial point.
            bnd_cond: str, default 'symmetric'. More commonly known as the padding or extension method used in discrete convolutions. All options supported by PyWavelets are valid.
                Most common examples are 'symmetric' (padding by mirroring edge values), 'zero' (padding with zeros), 'periodic' (wrapping values around as in circular convolution).
                Some padding methods can have unexpected effect on the wavelet coefficients at the edges.
                See https://pywavelets.readthedocs.io/en/latest/ref/signal-extension-modes.html for more details and all options.
        
        Attributes
        ----------
        :param moments: integer for number of vanishing moments.
            Default = Known for Daubechies, None for others
        
     '''     
       
    def __init__(self, domain_geometry, 
                       range_geometry=None, 
                       level = None,
                       wname = "haar",
                       axes = None,
                       **kwargs):

        # Correlation is different way of defining decomposition axes
        self.correlation = kwargs.get('correlation', None)
        
        if axes is None and len(domain_geometry.shape) > 1:
            if self.correlation in [None, 'All']:
                axes = None
            elif self.correlation.lower() in ["space", "spatial"]:
                axes = [i for i,l in enumerate(domain_geometry.dimension_labels) if l != 'channel']
            elif self.correlation.lower() in ["channels", "channel"]:
                axes = [i for i,l in enumerate(domain_geometry.dimension_labels) if l == 'channel']
            else:
                raise AttributeError(f"Unknown correlation type: '{self.correlation}'")
            if axes == []:
                raise AttributeError(f"Correlation set to '{self.correlation}' but the data only has '{domain_geometry.dimension_labels}' as possible dimensions")
        elif axes is not None and self.correlation is not None:
            warnings.warn(f"Decomposition axes '{axes}' take priority over correlation '{self.correlation}'. Both should not be used.", UserWarning)
        elif len(domain_geometry.shape) == 1 and self.correlation is not None:
            warnings.warn(f"Setting correlation '{self.correlation}' is not valid for 1D data.", UserWarning)

        # Convolution boundary condition i.e. padding method
        self.bnd_cond = kwargs.get('bnd_cond', 'symmetric')

        self.wname = wname
        self._wavelet = pywt.Wavelet(wname)
        self.moments = self._wavelet.vanishing_moments_psi
        self._trueAdj = kwargs.get('true_adjoint', True)
        if all([not self._wavelet.orthogonal, self._wavelet.biorthogonal, self._trueAdj]): # True adjoint for biorthogonal wavelet
            self._wavelet = self._getBiortFilters(wname)
        
        if level is None:
            # Default decomposition level is the theoretical maximum: log_2(min(input.shape)).
            # However, this is not always recommended and pywt should give a warning if the coarsest
            # scales are too small to be meaningful.
            level = pywt.dwtn_max_level(domain_geometry.shape, wavelet=self._wavelet, axes=axes)
        self.level = int(level)

        self._shapes = pywt.wavedecn_shapes(domain_geometry.shape, wavelet=self._wavelet, level=level, axes=axes, mode=self.bnd_cond)
        self.axes = axes
        self._slices = self._shape2slice()
        
        # Compute the correct wavelet domain size
        range_shape = np.array(domain_geometry.shape)
        if axes is None:
            axes = range(len(domain_geometry.shape))
        d = 'd'*len(axes) # Name of the diagonal element in unknown dimensional DWT
        for k in axes:
            range_shape[k] = self._shapes[0][k]
            for l in range(level):
                range_shape[k] += self._shapes[l+1][d][k]

        if range_geometry is None:
            range_geometry = domain_geometry.copy()

            # Update new size
            if hasattr(range_geometry, 'channels'):
                if range_geometry.channels > 1:
                    range_geometry.channels = range_shape[0]
                    range_shape = range_shape[1:] # Remove channels temporarily

            if len(range_shape) == 3:
                range_geometry.voxel_num_x = range_shape[2]
                range_geometry.voxel_num_y = range_shape[1]
                range_geometry.voxel_num_z = range_shape[0]
            elif len(range_shape) == 2:
                range_geometry.voxel_num_x = range_shape[1]
                range_geometry.voxel_num_y = range_shape[0]
            elif len(range_shape) == 1: # VectorGeometry is bit special
                range_geometry = VectorGeometry(range_shape[0])
            else:
                raise AttributeError(f"Spatial dimension of range_geometry can be at most 3. Now it is {len(range_shape)}!")
            
        elif (range_geometry.shape != range_shape).any():
            raise AttributeError(f"Size of the range geometry is {range_geometry.shape} but the size of the wavelet coefficient array must be {tuple(range_shape)}.")
                    
        super().__init__(domain_geometry=domain_geometry, range_geometry=range_geometry)


    def _shape2slice(self):
        """Helper function for turning shape of coefficients to slices"""
        shapes = self._shapes
        coeff_tmp = []
        coeff_tmp.append(np.empty(shapes[0]))

        for cd in shapes[1:]:
            subbs = dict((k, np.empty(v)) for k, v in cd.items())
            coeff_tmp.append(subbs)

        _, slices = pywt.coeffs_to_array(coeff_tmp, padding=0, axes=self.axes)
        return slices
    
    def _getBiortFilters(self, wname):
        """Helper function for creating a custom wavelet object.
        Using mirrored decomposition filters for reconstruction gives adjoint.
        This is only needed for biorthogonal wavelets."""
        fb = pywt.Wavelet(wname).filter_bank
        ifb = pywt.Wavelet(wname).inverse_filter_bank
        adj_filter_bank = fb[0:2] + ifb[2:4]
        wavelet = pywt.Wavelet(wname, filter_bank=adj_filter_bank)
        wavelet.orthogonal = False
        wavelet.biorthogonal = True
        return wavelet
        
    def direct(self, x, out = None):
        '''Forward operator -- decomposition -- analysis'''
        
        x_arr = x.as_array()
        
        coeffs = pywt.wavedecn(x_arr, wavelet=self._wavelet, level=self.level, axes=self.axes, mode=self.bnd_cond)

        Wx, _ = pywt.coeffs_to_array(coeffs, axes=self.axes)

        if out is None:
            ret = self.range_geometry().allocate()
            ret.fill(Wx)
            return ret
        else:
            out.fill(Wx) 
    
    def adjoint(self, Wx, out = None):
        '''Adjoint operator -- reconstruction -- synthesis'''
                      
        Wx_arr = Wx.as_array()
        coeffs = pywt.array_to_coeffs(Wx_arr, self._slices)

        x = pywt.waverecn(coeffs, wavelet=self._wavelet, axes=self.axes, mode=self.bnd_cond)

        # Need to slice the output in case original size is of odd length
        org_size = tuple(slice(i) for i in self.domain_geometry().shape)

        if out is None:
            ret = self.domain_geometry().allocate()
            ret.fill(x[org_size])
            return ret
        else:
            out.fill(x[org_size])
        
    def calculate_norm(self):
        if self._wavelet.orthogonal:
            norm = 1.0
        else:
            norm = LinearOperator.calculate_norm(self)
        return norm
