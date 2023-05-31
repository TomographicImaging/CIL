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

from cil.framework import Processor, DataContainer
import numpy as np
import weakref
import logging
from numba import jit, prange



@jit(parallel=True, nopython=True)
def numba_apply_normalisation_default(arr_in, arr_out, offset, scale, num_proj):
    #numba supports operations on multidimensional arrays of the same size so offset and scale can be float or images
    for i in prange(num_proj):
        arr_out[i] = (arr_in[i] - offset) * scale


@jit(parallel=True, nopython=True)
def numba_apply_normalisation_default_inplace(arr_in, offset, scale, num_proj):
    #numba supports operations on multidimensional arrays of the same size so offset and scale can be float or images
    for i in prange(num_proj):
        arr_in[i] -= offset 
        arr_in[i] *= scale


class Normaliser(Processor):

    def __init__(self, flat_field=None, dark_field=None, method='default', tolerance=1e-6):
        """
        Acquisition Data normalisation applying corrections as:

        (data - darkfield) / (flatfield - darkfield)

        
        Parameters
        ----------
        flat_field: float, ndarray, optional
            the normalisation arrays, stack of arrays or value
        dark_field: float, ndarray
            The normalisation arrays, stack of arrays or value
        method: str, optional

            'default' - applies single correction to all projections 
            requires a float or an ndarray matching the panel size
            
            'mean' -  applies single correction to all projections 
            requires stack of ndarrays matching panel size

            'stack' - applies a different correction to each projection
            apply one correction per projection
            requires a list of floats or a stack of ndarrays matching panel size

        """

        if method not in ['default', 'mean', 'stack']:
            raise ValueError("'method' not recognised")

        if flat_field is None and dark_field is None:
            logging.warning("No normalisation images/values provided")

        kwargs = {
                  '_darkfield'  : 0.0,
                  '_flatfield'  : 1.0,
                  '_method' : method,
                  '_tolerance' : tolerance

                  }

        super(Normaliser, self).__init__(**kwargs)


        #set up
        if isinstance(flat_field, np.ndarray):
            flat_field = np.asarray(flat_field,dtype=np.float32)

        if isinstance(dark_field, np.ndarray):
            dark_field = np.asarray(dark_field,dtype=np.float32)

        if dark_field is not None:

            if method == 'default':
                
                if isinstance(dark_field, np.ndarray):
                    if dark_field.ndim > 2:
                        raise ValueError("'default' mode normalisation requires a single value or image only")
                else:
                    try:
                        float(dark_field)
                    except TypeError:
                        raise TypeError("'default' mode normalisation requires a single value or image only")
                
                self._darkfield = dark_field


            if method == 'mean':
                self._darkfield =  np.mean(dark_field, axis=0)
            else:
                self._darkfield =  dark_field


        if flat_field is not None:
            if method == 'default':

                if isinstance(flat_field, np.ndarray):
                    if flat_field.ndim > 2:
                        raise ValueError("'default' mode normalisation requires a single value or image only")
                else:
                    try:
                        float(flat_field)
                    except TypeError:
                        raise TypeError("'default' mode normalisation requires a single value or image only")


            if method == 'mean':
                self._flatfield =  np.mean(flat_field, axis=0)
            else:
                self._flatfield =  flat_field


    def set_input(self, dataset):
        """
        Set the input data to the processor

        Parameters
        ----------
        input : DataContainer
            The input DataContainer
        """

        if issubclass(type(dataset), (DataContainer, np.ndarray)):
            if self.check_input(dataset):
                self.__dict__['input'] = weakref.ref(dataset)
                self.__dict__['shouldRun'] = True
            else:
                raise ValueError('Input data not compatible')
        else:
            raise TypeError("Input type mismatch: got {0} expecting {1}"\
                            .format(type(dataset), DataContainer))
        

    def check_input(self, dataset):

        # if dataset.ndim > 3:
        #     shape_expected = dataset.shape[-2,-1]
        # else:
        #     shape_expected = dataset.shape[-1]

        # if self._method == 'stack':
        #     if dataset.shape[0] != self._scale.shape[0]:
        #         raise ValueError("method 'stack' expects the normalisation shape to be the same as the data shape")       
        # else:        
        #     if len(self._scale.shape) > 1:
        #      if self._scale.shape != shape_expected:
        #          raise ValueError("Shape mismatch")
             
        return True


    def process(self, out=None):

        projections = self.get_input()
        return_flag = False

        if out is None:
            return_flag = True
            out = projections.geometry.allocate(None)
        else:
            if out.shape != projections.shape or out.dtype != projections.dtype:
                raise AttributeError("out not compatible with input data")
            
        if self._method == 'default' or self._method=='mean':
                self.apply_normalisation_default(projections.array, out.array, self._darkfield, self._flatfield)
        else:
            raise NotImplementedError("stack not implemented yet")


        if return_flag:
            return out
            


    @staticmethod
    def apply_normalisation_default(arr_in, arr_out, dark, flat):
        """
        Applies the default normalisation on numpy arrays      
        """
        shape_orig = arr_in.shape

        if isinstance(flat, np.ndarray):
            num_chunks = arr_in.size / flat.size
            arr_in.shape = (num_chunks, *flat.shape)

        elif isinstance(dark, np.ndarray):
            num_chunks = arr_in.size / dark.size
            arr_in.shape = (num_chunks, *dark.shape)

        elif arr_in.ndim == 4:
                num_chunks = arr_in[0] * arr_in[1]
                arr_in.shape = (num_chunks, *arr_in.shape[2::])
        else:
            num_chunks = arr_in.shape[0]


        with np.errstate(divide='ignore', invalid='ignore'):
            scale_img = 1 / (flat - dark)

        if id(arr_in) == id(arr_out):
            numba_apply_normalisation_default_inplace(arr_in, dark, scale_img, num_chunks)
        else:
            arr_out.shape = arr_in.shape
            numba_apply_normalisation_default_inplace(arr_in, arr_out, dark, scale_img, num_chunks)
            arr_out.shape = shape_orig

        #reset shape
        arr_in.shape = shape_orig





