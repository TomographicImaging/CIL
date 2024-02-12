# -*- coding: utf-8 -*-
#  Copyright 2021 United Kingdom Research and Innovation
#  Copyright 2021 The University of Manchester
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

from cil.framework import AcquisitionData, ImageGeometry, check_order_for_engine
import importlib
import weakref

class Reconstructor(object):
    
    """ Abstract class representing a reconstructor 
    """

    supported_backends = ['tigre']
    
    #_input is a weakreference object
    @property
    def input(self):
        if self._input() is None:
            raise ValueError("Input has been deallocated")
        else:
            return self._input()


    @property
    def acquisition_geometry(self):
        return self._acquisition_geometry


    @property
    def image_geometry(self):
        return self._image_geometry


    @property
    def backend(self):
        return self._backend


    def __init__(self, input, image_geometry=None, backend='tigre'):


        if not issubclass(type(input), AcquisitionData):
            raise TypeError("Input type mismatch: got {0} expecting {1}"
                            .format(type(input), AcquisitionData))

        self._acquisition_geometry = input.geometry.copy()
        self._configure_for_backend(backend)
        self.set_image_geometry(image_geometry)
        self.set_input(input)


    def set_input(self, input):
        """
        Update the input data to run the reconstructor on. The geometry of the dataset must be compatible with the reconstructor.

        Parameters
        ----------
        input : AcquisitionData
            A dataset with a compatible geometry

        """
        if input.geometry != self.acquisition_geometry:
            raise ValueError ("Input not compatible with configured reconstructor. Initialise a new reconstructor with this geometry")
        else:
            self._input = weakref.ref(input)


    def set_image_geometry(self, image_geometry=None):
        """
        Sets a custom image geometry to be used by the reconstructor

        Parameters
        ----------
        image_geometry : ImageGeometry, default used if None
            A description of the area/volume to reconstruct
        """
        if image_geometry is None:
            self._image_geometry = self.acquisition_geometry.get_ImageGeometry()
        elif issubclass(type(image_geometry), ImageGeometry):
            self._image_geometry = image_geometry.copy()
        else:
            raise TypeError("ImageGeometry type mismatch: got {0} expecting {1}"\
                                .format(type(input), ImageGeometry))   
           

    def _configure_for_backend(self, backend='tigre'):
        """
        Configures the class for the right engine. Checks the dataorder.
        """        
        if backend not in self.supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self.supported_backends))

        if not check_order_for_engine(backend, self.acquisition_geometry):
            raise ValueError("Input data must be reordered for use with selected backend. Use input.reorder{'{0}')".format(backend))

        #set ProjectionOperator class from backend
        try:
            module = importlib.import_module('cil.plugins.'+backend)
        except ImportError:
            if backend == 'tigre':
                raise ImportError("Cannot import the {} plugin module. Please install TIGRE or select a different backend".format(self.backend))
            if backend == 'astra':
                raise ImportError("Cannot import the {} plugin module. Please install CIL-ASTRA or select a different backend".format(self.backend))

        self._PO_class = module.ProjectionOperator
        self._backend = backend


    def reset(self):
        """
        Resets all optional configuration parameters to their default values
        """
        raise NotImplementedError()


    def run(self, out=None, verbose=1):
        """
        Runs the configured recon and returns the reconstruction

        Parameters
        ----------
        out : ImageData, optional
           Fills the referenced ImageData with the reconstructed volume and suppresses the return
        
        verbose : int, default=1
           Contols the verbosity of the reconstructor. 0: No output is logged, 1: Full configuration is logged

        Returns
        -------
        ImageData
            The reconstructed volume. Suppressed if `out` is passed
        """

        raise NotImplementedError()


    def _str_data_size(self):

        repres = "\nInput Data:\n"
        for dim in  zip(self.acquisition_geometry.dimension_labels,self.acquisition_geometry.shape):
            repres += "\t" + str(dim[0]) + ': ' + str(dim[1])+'\n'

        repres += "\nReconstruction Volume:\n"
        for dim in zip(self.image_geometry.dimension_labels,self.image_geometry.shape):
            repres += "\t" + str(dim[0]) + ': ' + str(dim[1]) +'\n'

        return repres