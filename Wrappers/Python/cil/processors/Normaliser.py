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

from cil.framework import Processor, DataContainer, AcquisitionData,\
 AcquisitionGeometry, ImageGeometry, ImageData
import numpy as np
import weakref
import logging

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
                  '_offset'  : False,
                  '_scale'  : False,
                  '_method' : method
                  }

        super(Normaliser, self).__init__(**kwargs)

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
                
                self._offset = dark_field

            elif method == 'mean':
                self._offset =  np.mean(dark_field, axis=0)

            else:
                self._offset =  dark_field


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

            elif method == 'mean':
                self._flatfield =  np.mean(flat_field, axis=0)

            else:
                self._flatfield =  flat_field

            if self._offset:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._scale = 1.0 /(flat_field - self._offset)
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    self._scale = 1.0 / flat_field

            np.nan_to_num(self._scale,posinf=tolerance, neginf=tolerance)


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

        if out is None:
            flag = False
            if self._offset:
                data = np.subtract(projections,self._offset,out=data)
                flag = True

            if self._scale:
                if flag:
                    data *= self._scale
                else:
                    data = projections * self._scale 
            
            return data

        else:
            flag = False
            if self._offset:
                if id(out) == id(projections):
                    projections -= self._offset
                else:
                    out.fill(projections - self._offset)
                flag = True
                    
            if self._scale:
                if flag:
                    out *= self._scale
                else:
                    if id(out) == id(projections):
                        projections *= self._scale
                    else:
                        out.fill(projections*self._scale)