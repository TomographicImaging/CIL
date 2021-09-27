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
from cil.framework import AcquisitionGeometry
from cil.reconstructors import Reconstructor
from scipy.fft import fftfreq
from cil.plugins.tigre import ProjectionOperator

import numpy as np
import ctypes, platform
from ctypes import util

if platform.system() == 'Linux':
    dll = 'libcilacc.so'
elif platform.system() == 'Windows':
    dll_file = 'cilacc.dll'
    dll = util.find_library(dll_file)
elif platform.system() == 'Darwin':
    dll = 'libcilacc.dylib'
else:
    raise ValueError('Not supported platform, ', platform.system())

cilacc = ctypes.cdll.LoadLibrary(dll)

c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)

cilacc.filter_projections_avh.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the data array 
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the filter array
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the weights array
                                  ctypes.c_int16, #order of the fft
                                  ctypes.c_long, #num_proj
                                  ctypes.c_long, #pix_v
                                  ctypes.c_long] #pix_x

cilacc.filter_projections_vah.argtypes = [ctypes.POINTER(ctypes.c_float),  # pointer to the data array 
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the filter array
                                  ctypes.POINTER(ctypes.c_float),  # pointer to the weights array
                                  ctypes.c_int16, #order of the fft
                                  ctypes.c_long, #pix_v
                                  ctypes.c_long, #num_proj
                                  ctypes.c_long] #pix_x

class FBP(Reconstructor):

    @property
    def filter(self):
        return self.__filter

    @filter.setter
    def filter(self, val):
        self.set_filter(val)

    @property
    def filter_inplace(self):
        return self.__filter_inplace

    @filter_inplace.setter
    def filter_inplace(self, val):
        self.set_filter_inplace(val)

    @property
    def fft_order(self):
        return self.__fft_order

    @fft_order.setter
    def fft_order(self, val):
        self.set_fft_order(val)

    def __init__ (self,input):
        """
        Creates an FBP/FDK reconstructor with a 'ram-lak' filter.

        The reconstructor can be customised using:
        self.set_input()
        self.set_image_geometry()
        self.set_backend()
        self.set_filter()
        self.get_filter()
        self.set_fft_order()
        self.set_filter_inplace()
        """
        #additional check
        if 'channel' in input.dimension_labels:
            raise ValueError("Input data cannot be multi-channel")

        #call parent initialiser
        super(FBP, self).__init__(input)

        #define defaults
        self.__filter = 'ram-lak' 
        self.__fft_order =self.__min_fft_order()
        self.__filter_inplace = False

    def set_filter_inplace(self, inplace):
        """
        False (default) will allocate temporary memory for filtered projections.
        True will filter projections in-place.

        :param inplace: Sets the inplace filtering of projections.
        :type inplace: boolian
        """
        if type(inplace) is bool:
            self.__filter_inplace= inplace
        else:
            raise TypeError("set_filter_inplace expected a boolian. Got {}".format(type(inplace)))
        
    def __min_fft_order(self):
        min_order = 0
        while 2**min_order < self.input.geometry.pixel_num_h * 2:
            min_order+=1

        min_order = max(8, min_order)
        return min_order

    def set_fft_order(self, order):
        """
        The width of the fourier transform N=2^order. Higher orders yield more accurate results.

        The default is the max of 8, or power-of-2 greater than detector width * 2.

        :param set_fft_order: The width of the fft N=2^order 
        :type set_fft_order: int
        """
        try:
            fft_order = int(order)

        except TypeError:
            raise TypeError("fft order expected type `int`. Got{}".format(type(order)))
        
        min_order = self.__min_fft_order()
        if fft_order < min_order:
            raise ValueError("Minimum fft width 2^order is order = {0}. Got{1}".format(min_order,order))

        if self.filter=='custom' and fft_order != self.fft_order:
            print("Filter length changed - resetting filter array to ram-lak")
            self.__filter='ram-lak'
            del self.__filter_array
        
        self.__fft_order =fft_order

    def set_filter(self, filter):
        """
        Set the filter used by the reconstruction. This is set to 'ram-lak' by default.

        A custom filter can be set of length (N) 2^self.fft_order

        The indices of the array are interpreted as:
        0 The DC frequency component
        1:N/2 positive frequencies
        N/2:N-1 negative frequencies

        :param filter: The filter to be applied. Can be a string from: 'ram-lak' or a numpy array.
        :type filter: string, numpy.ndarray
        """

        if filter in ['ram-lak']:
            self.__filter == filter
        elif type(filter)==np.ndarray:
            try:
                filter_array = np.asarray(filter,dtype=np.float32).reshape(2**self.fft_order) 
                self.__filter_array = filter_array.copy()
                self.__filter = 'custom'
            except ValueError:
                raise ValueError("Custom filter not compatible with input.")
        else:
            raise ValueError("Filter not recognised")
            
    def get_filter_array(self):
        """
        Returns the filter in used in the frequency domain. The array can be modified and passed back using set_filter()

        The filter length N is 2^self.fft_order.

        The indices of the array are interpreted as:
        0 The DC frequency component
        1:N/2 positive frequencies
        N/2:N-1 negative frequencies

        :return: An array containing the filter values
        :rtype: numpy.ndarray
        """
        if self.filter == 'ram-lak':
            filter_length = 2**self.fft_order
            freq = fftfreq(filter_length)
            filter = np.asarray( [ np.abs(2*el) for el in freq ] ,dtype=np.float32)
        elif self.filter == 'custom':
            filter = self.__filter_array
        return filter

    def __calculate_weights_fdk(self):
        """
        Calculates the pre-weighting used for FDK reconstruction.

        :return: A single image containing the weights perpixel
        :rtype: numpy.ndarray        
        """
        ag = self.input.geometry
        xv = np.arange(-(ag.pixel_num_h -1)/2,(ag.pixel_num_h -1)/2 + 1,dtype=np.float32) * ag.pixel_size_h
        yv = np.arange(-(ag.pixel_num_v -1)/2,(ag.pixel_num_v -1)/2 + 1,dtype=np.float32) * ag.pixel_size_v
        (yy, xx) = np.meshgrid(xv, yv)

        principal_ray_length = ag.dist_source_center + ag.dist_center_detector
        scaling =  ag.magnification * (2 * np.pi/ len(ag.angles)) / ( 4 * ag.pixel_size_h ) 
        weights = scaling * principal_ray_length / np.sqrt((principal_ray_length ** 2 + xx ** 2 + yy ** 2))
        return weights

    def __calculate_weights_fbp(self):
        """
        Calculates the pre-weighting used for FBP reconstruction.

        :return weights: A single image containing the weights perpixel
        :rtype weights: numpy.ndarray
        """
        ag = self.input.geometry
        weights = np.ones(ag.pixel_num_h,ag.pixel_num_v)
        return weights

    def __pre_filtering(self,acquistion_data):
        """
        Filters and weights the projections inplace. The filtering
        can be configured by setting the properties:
        self.fft_order
        self.filter

        :param acquistion_data: The projections to be filtered
        :type acquistion_data: AcquisitionData
    '''
        """
        nda = acquistion_data.as_array()

        filter=self.get_filter_array()

        if AcquisitionGeometry.CONE:
            weights = self.__calculate_weights_fdk()
        else:
            weights = self.__calculate_weights_fbp()

        #call ext function
        data_ptr = nda.ctypes.data_as(c_float_p)
        filter_ptr = filter.ctypes.data_as(c_float_p)
        weights_ptr = weights.ctypes.data_as(c_float_p)

        ag = self.input.geometry
        if ag.dimension_labels == ('angle','vertical','horizontal'):   
            cilacc.filter_projections_avh(data_ptr, filter_ptr, weights_ptr, self.fft_order, *self.input.shape)
        elif ag.dimension_labels == ('angle','horizontal'): 
            cilacc.filter_projections_vah(data_ptr, filter_ptr, weights_ptr, self.fft_order, 1, *self.input.shape) 
        else:
            raise ValueError ("Gemma says no")

        acquistion_data.fill(nda)

    def run(self, out=None):
        """
        Run the configured FBP/FDK reconstruction.

        :param out: Fills the referenced array with the FBP/FDK reconstruction of the acquisition data
        :type out: ImageData
        :return: returns the FBP/FDK reconstruction of the AcquisitionData
        :rtype: ImageData
        """

        if self.filter_inplace is False:
            proj_filtered = self.input.copy()
        else:
            proj_filtered = self.input

        self.__pre_filtering(proj_filtered)
            
        operator = ProjectionOperator(self.image_geometry,self.input.geometry,adjoint_method='FDK')
        
        if out == None:
            return operator.adjoint(proj_filtered)
        else:
            operator.adjoint(proj_filtered, out = out)


