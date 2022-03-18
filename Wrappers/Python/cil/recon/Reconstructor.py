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

from cil.framework import AcquisitionData, ImageGeometry, DataOrder
import weakref

class Reconstructor(object):
    
    """ Abstract class representing a reconstructor 
    """

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
        if self._image_geometry is None:
            return self.acquisition_geometry.get_ImageGeometry()
        else:
            return self._image_geometry


    @property
    def backend(self):
        return self._backend


    def __init__(self, input, image_geometry=None):
        self._backend = 'tigre'

        if not issubclass(type(input), AcquisitionData):
            raise TypeError("Input type mismatch: got {0} expecting {1}"
                            .format(type(input), AcquisitionData))

        if not DataOrder.check_order_for_engine(self.backend, input.geometry):
            raise ValueError("Input data must be reordered for use with selected backed. Use input.reorder{'{0}')".format(self._backend))

        self._acquisition_geometry = input.geometry.copy()
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
            self._image_geometry = None
        elif issubclass(type(image_geometry), ImageGeometry):
            self._image_geometry = image_geometry.copy()
        else:
            raise TypeError("ImageGeometry type mismatch: got {0} expecting {1}"\
                                .format(type(input), ImageGeometry))   
           

    def set_backend(self, backend='tigre'):
        """
        Sets the backend used for the foward/backward projectors. Currently only TIGRE is supported
        
        Parameters
        ----------
        backend: string
            Set the backend to TIGRE 'tigre'
        """        
        supported_backends = ['tigre']
        if backend not in supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}", supported_backends)
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