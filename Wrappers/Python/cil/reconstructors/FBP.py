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
from cil.framework import cilacc
from cil.framework import AcquisitionGeometry
from cil.reconstructors import Reconstructor
from scipy.fft import fftfreq
from cil.plugins.tigre import ProjectionOperator

import numpy as np
import ctypes, platform
from ctypes import util


c_float_p = ctypes.POINTER(ctypes.c_float)
c_double_p = ctypes.POINTER(ctypes.c_double)

try:
    cilacc.filter_projections_avh
    has_ipp = True
except:
    has_ipp = False

if has_ipp:
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

class FBP_base(Reconstructor):

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

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, val):
        self.__weights = val

    def __init__ (self,input):
        """
        The initialiser for abstract base class::FBP_base

        :param input: The input data to reconstruct. The reconstructor is set-up based on the geometry of the data. 
        :type input: AcquisitionData
        """
        if has_ipp == False:
            raise ImportError("IPP libraries not found. Cannot use CIL FBP")

        #call parent initialiser
        super(FBP_base, self).__init__(input)
        
        #additional check
        if 'channel' in input.dimension_labels:
            raise ValueError("Input data cannot be multi-channel")

        #define defaults
        self.__fft_order = self.__default_fft_order()
        self.set_filter('ram-lak')
        self.__filter_inplace = False
        self.__weights = None


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


    def __default_fft_order(self):
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
        
        min_order = self.__default_fft_order()
        if fft_order < min_order:
            raise ValueError("Minimum fft width 2^order is order = {0}. Got{1}".format(min_order,order))

        if fft_order != self.fft_order:
            self.__fft_order =fft_order

            if self.filter=='custom':
                print("Filter length changed - resetting filter array to ram-lak")
                self.set_filter('ram-lak')
            else:
                self.set_filter(self.__filter)
        
 
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
            self.__filter = filter

            if filter == 'ram-lak':
                filter_length = 2**self.fft_order
                freq = fftfreq(filter_length)
                self.__filter_array = np.asarray( [ np.abs(2*el) for el in freq ] ,dtype=np.float32)

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
    
        return self.__filter_array


    def calculate_weights(self):
        return NotImplementedError


    def pre_filtering(self,acquistion_data):
        """
        Filters and weights the projections inplace. The filtering
        can be configured by setting the properties:
        self.fft_order
        self.filter

        :param acquistion_data: The projections to be filtered
        :type acquistion_data: AcquisitionData
    '''
        """
        if self.weights is None or self.weights.shape[0] != acquistion_data.geometry.pixel_num_v:
            self.calculate_weights(acquistion_data.geometry)

        filter_array = self.get_filter_array()

        if self.weights.shape[1] != acquistion_data.shape[-1]: #horizontal
            raise ValueError("Weights not compatible")

        if filter_array.size != 2**self.fft_order:
            raise ValueError("Filter not compatible")

        #call ext function
        data_ptr = acquistion_data.array.ctypes.data_as(c_float_p)
        filter_ptr = filter_array.ctypes.data_as(c_float_p)
        weights_ptr = self.__weights.ctypes.data_as(c_float_p)

        ag = acquistion_data.geometry
        if ag.dimension_labels == ('angle','vertical','horizontal'):   
            cilacc.filter_projections_avh(data_ptr, filter_ptr, weights_ptr, self.fft_order, *acquistion_data.shape)
        elif ag.dimension_labels == ('angle','horizontal'): 
            cilacc.filter_projections_vah(data_ptr, filter_ptr, weights_ptr, self.fft_order, 1, *acquistion_data.shape) 
        elif ag.dimension_labels == ('vertical','horizontal'): 
            cilacc.filter_projections_avh(data_ptr, filter_ptr, weights_ptr, self.fft_order, 1, *acquistion_data.shape) 
        else:
            raise ValueError ("The data is not in a compatible order. Try reordering the data with data.reorder({})".format(self.backend))


    def run(self, out=None):
        NotImplementedError


class FDK(FBP_base):

    def __init__ (self,input):
        """
        Creates an FDK reconstructor based on your acquisition data with a 'ram-lak' filter.

        The reconstructor can be customised using:
        self.set_input()
        self.set_image_geometry()
        self.set_backend()
        self.set_filter()
        self.get_filter_array()
        self.set_fft_order()
        self.set_filter_inplace()

        :param input: The input data to reconstruct. The reconstructor is set-up based on the geometry of the data. 
        :type input: AcquisitionData
        """
        #call parent initialiser
        super(FDK, self).__init__(input)
        if  input.geometry.geom_type != AcquisitionGeometry.CONE:
            raise TypeError("This reconstructor is for cone-beam data only.")

        
    def calculate_weights(self, acquistion_geometry):
        """
        Calculates the pre-weighting used for FDK reconstruction.   
        """
        ag = acquistion_geometry
        xv = np.arange(-(ag.pixel_num_h -1)/2,(ag.pixel_num_h -1)/2 + 1,dtype=np.float32) * ag.pixel_size_h
        yv = np.arange(-(ag.pixel_num_v -1)/2,(ag.pixel_num_v -1)/2 + 1,dtype=np.float32) * ag.pixel_size_v
        (yy, xx) = np.meshgrid(xv, yv)

        principal_ray_length = ag.dist_source_center + ag.dist_center_detector
        scaling =  ag.magnification * (2 * np.pi/ ag.num_projections) / ( 4 * ag.pixel_size_h ) 
        self.weights = scaling * principal_ray_length / np.sqrt((principal_ray_length ** 2 + xx ** 2 + yy ** 2))


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

        self.pre_filtering(proj_filtered)
        operator = ProjectionOperator(self.image_geometry,self.input.geometry,adjoint_weights='FDK')
        
        if out is None:
            return operator.adjoint(proj_filtered)
        else:
            operator.adjoint(proj_filtered, out = out)


class FBP(FBP_base):

    @property
    def by_slice(self):
        return self.__by_slice

    @by_slice.setter
    def by_slice(self, val):
        self.set_by_slice(val)

    def set_split_processing(self, slice_data):
        """
        False (default) will process the data in a single call.
        True will process the data slice-by-slice, this will reduce memory use, but increase computation time.

        it can only be used on simple data-geometries
        :param slice_data: Sets the split processing of the data.
        :type inplace: boolian
        """
        if type(slice_data) is not bool:
            raise TypeError("set_split_processing expected a boolian. Got {}".format(type(slice_data)))

        if slice_data == True:
            if self.input.geometry.dimension != '3D':
                print("Only 3D data can be processed in chunks, setting slice_data to `False`")
                slice_data = False

            if self.input.geometry.system_description == 'advanced':
                print("Only simple and offset geometries can be processed in chunks, setting slice_data to `False`")
                slice_data = False

            if self.input.geometry.get_ImageGeometry() != self.image_geometry:
                print("Only default image geometries can be processed in chunks, setting slice_data to `False`")
                slice_data = False

        self.__by_slice= slice_data


    def __init__ (self,input):
        """
        Creates an FBP reconstructor based on your acquisition data with a 'ram-lak' filter.

        The reconstructor can be customised using:
        self.set_input()
        self.set_image_geometry()
        self.set_backend()
        self.set_filter()
        self.get_filter_array()
        self.set_fft_order()
        self.set_filter_inplace()
        self.set_split_processing()


        :param input: The input data to reconstruct. The reconstructor is set-up based on the geometry of the data. 
        :type input: AcquisitionData
        """
        super(FBP, self).__init__(input)


        if  input.geometry.geom_type != AcquisitionGeometry.PARALLEL:
            raise TypeError("This reconstructor is for parallel-beam data only.")

        self.set_split_processing(False)
         
    def calculate_weights(self, acquistion_geometry):
        """
        Calculates the weights used for FBP reconstruction.     
        """
        ag = acquistion_geometry
        weight = (2 * np.pi/ ag.num_projections) / ( 4 * ag.pixel_size_h ) 
 
        self.weights = np.full((ag.pixel_num_v,ag.pixel_num_h),weight,dtype=np.float32)


    def run(self, out=None):
        """
        Run the configured FBP/FDK reconstruction.

        :param out: Fills the referenced array with the FBP/FDK reconstruction of the acquisition data
        :type out: ImageData
        :return: returns the FBP/FDK reconstruction of the AcquisitionData
        :rtype: ImageData
        """
        
        if self.by_slice:
            if out is None:
                ret = self.image_geometry.allocate()
            else:
                ret = out

            data_slice = self.input.get_slice(vertical=0)               
            ag_slice = data_slice.geometry
            ig_slice = self.image_geometry.get_slice(vertical=0)
            operator = ProjectionOperator(ig_slice,ag_slice)

            if self.filter_inplace:
                self.pre_filtering(self.input)

                for i in range(self.image_geometry.shape[0]):
                    data_slice.fill(self.input.get_slice(vertical=i))
                    ret.array[i,:,:] = operator.adjoint(data_slice).array[:,:]
            else:
                for i in range(self.image_geometry.shape[0]):
                    data_slice.fill(self.input.get_slice(vertical=i))
                    self.pre_filtering(data_slice)
                    ret.array[i,:,:] = operator.adjoint(data_slice).array[:,:]

            if out is None:
                return ret

        else:

            if self.filter_inplace is False:
                proj_filtered = self.input.copy()
            else:
                proj_filtered = self.input

            self.pre_filtering(proj_filtered)

            operator = ProjectionOperator(self.image_geometry,self.input.geometry)

            if out is None:
                return operator.adjoint(proj_filtered)
            else:
                operator.adjoint(proj_filtered, out = out)