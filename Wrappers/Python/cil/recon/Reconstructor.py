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

    #__input is a weakreference object
    @property
    def input(self):
        if self.__input() is None:
            raise ValueError("Input has been deallocated")
        else:
            return self.__input()

    @input.setter
    def input(self, val):
        self.set_input(val)

    @property
    def acquisition_geometry(self):
        return self.input.geometry

    @property
    def image_geometry(self):
        if self.__image_geometry is None:
            return self.acquisition_geometry.get_ImageGeometry()
        else:
            return self.__image_geometry

    @image_geometry.setter
    def image_geometry(self, val):
        self.set_image_geometry(val)

    @property
    def backend(self):
        return self.__backend

    @backend.setter
    def backend(self, val):
        self.set_backend(val)


    def __init__(self, input):
        self.__backend = 'tigre'

        if not issubclass(type(input), AcquisitionData):
            raise TypeError("Input type mismatch: got {0} expecting {1}"
                            .format(type(input), AcquisitionData))

        if not DataOrder.check_order_for_engine(self.backend, input.geometry):
            raise ValueError("Input data must be reordered for use with selected backed. Use input.reorder{'{0}')".format(self.__backend))

        self.__input = weakref.ref(input)
        self.__image_geometry = None

    
    def set_input(self, input):
        """
        Update the data to run the reconstructor on. The new data must
        have the same geometry as the initial data used to configure the reconstructor.

        :param input: A dataset with the same geometry
        :type input: AcquisitionData
        """
        if input.geometry != self.acquisition_geometry:
            raise ValueError ("Input not compatible with configured reconstructor. Initialise a new reconstructor with this geometry")
        else:
            self.__input = weakref.ref(input)


    def set_image_geometry(self, image_geometry):
        """
        :param image_geometry: Set the ImageGeometry of the reconstructor
        :type image_geometry: ImageGeometry
        """
        if image_geometry is None:
            self.__image_geometry = None
        elif issubclass(type(image_geometry), ImageGeometry):
            self.__image_geometry = image_geometry.copy()
        else:
            raise TypeError("ImageGeometry type mismatch: got {0} expecting {1}"\
                                .format(type(input), ImageGeometry))   
           

    def set_backend(self, backend):
        """
        :param backend: Set the backend used for the foward/backward projectors
        :type backend: string, 'tigre'
        """
        supported_backends = ['tigre']
        if backend not in supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}", supported_backends)
        self.__backend = backend


    def run(self):
        raise NotImplementedError()



