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

import importlib
from typing import Literal, Optional
import weakref
from cil.framework import AcquisitionData, ImageGeometry, DataOrder, ImageData, AcquisitionGeometry
from abc import ABCMeta

class Reconstructor(metaclass=ABCMeta):
    """Abstract class for a tomographic reconstructor.

    Parameters
    ----------
    input
        Input data for reconstruction
    image_geometry
        Image geometry of input data, by default None
    backend
        Engine backend used for reconstruction, by default "tigre"

    Raises
    ------
    TypeError
        Raised if input data is not an AcquisitionData class
    """

    def __init__(self, input:AcquisitionData, image_geometry:Optional[ImageGeometry]=None, backend:Literal["tigre"]="tigre"):
        if not issubclass(type(input), AcquisitionData):
            raise TypeError("Input type mismatch: got {0} expecting {1}"
                            .format(type(input), AcquisitionData))

        self._acquisition_geometry = input.geometry.copy()
        self._configure_for_backend(backend)
        self.set_image_geometry(image_geometry)
        self.set_input(input)

    supported_backends = ["tigre"]

    @property
    def input(self) -> AcquisitionData:
        """Get the input data used for reconstruction.

        Returns
        -------
        AcquisitionData
            Input data used for reconstruction.

        Raises
        ------
        ValueError
            Raised if input has been deallocated.
        """
        #_input is a weakreference object
        if self._input() is None:
            raise ValueError("Input has been deallocated")
        else:
            return self._input()

    @property
    def acquisition_geometry(self) -> AcquisitionGeometry:
        """Get the acquisition geometry used for reconstruction.

        Returns
        -------
        AcquisitionGeometry
            The acquisition geometry used for reconstruction
        """
        return self._acquisition_geometry

    @property
    def image_geometry(self) -> ImageGeometry:
        """Get the image geometry used for reconstruction.

        Returns
        -------
        ImageGeometry
            Image geometry used for reconstruction
        """
        return self._image_geometry

    @property
    def backend(self) -> str:
        """Get the backend engine used for reconstruction.

        Returns
        -------
        str
            Backend engine used for reconstruction
        """
        return self._backend

    def set_input(self, input:AcquisitionData) -> None:
        """
        Update the input data to run the reconstructor on. The geometry of the dataset must be compatible with the reconstructor.

        Parameters
        ----------
        input
            A dataset with a compatible geometry
        """
        if input.geometry != self.acquisition_geometry:
            raise ValueError ("Input not compatible with configured reconstructor. Initialise a new reconstructor with this geometry")
        else:
            self._input = weakref.ref(input)

    def set_image_geometry(self, image_geometry:Optional[ImageGeometry]=None):
        """Set a custom image geometry to be used by the reconstructor.

        Parameters
        ----------
        image_geometry
            A description of the area/volume to reconstruct
        """
        if image_geometry is None:
            self._image_geometry = self.acquisition_geometry.get_ImageGeometry()
        elif issubclass(type(image_geometry), ImageGeometry):
            self._image_geometry = image_geometry.copy()
        else:
            raise TypeError("ImageGeometry type mismatch: got {0} expecting {1}"\
                                .format(type(input), ImageGeometry))

    def reset(self) -> None:
        """Reset all optional configuration parameters to their default values."""
        raise NotImplementedError()

    def run(self, out:Optional[ImageData]=None, verbose:int=1) -> Optional[ImageData]:
        """Run. the configured recon and returns the reconstruction.

        Parameters
        ----------
        out : ImageData, optional
           Fills the referenced ImageData with the reconstructed volume and suppresses the return

        verbose : int, default=1
           Controls the verbosity of the reconstructor. 0: No output is logged, 1: Full configuration is logged

        Returns
        -------
        ImageData
            The reconstructed volume. Suppressed if `out` is passed
        """
        raise NotImplementedError()

    def _configure_for_backend(self, backend:Literal["tigre"]="tigre") -> None:
        """Configure the reconstructor class for the right engine, checking the dataorder.

        Parameters
        ----------
        backend
            Engine backend, by default "tigre"

        Raises
        ------
        ValueError
            Raised if an unsupported backend is passed.
        ValueError
            Raised if the input data must be reordered for the selected backend.
        ImportError
            Raised if the selected backend cannot be imported.
        """
        if backend not in self.supported_backends:
            raise ValueError("Backend unsupported. Supported backends: {}".format(self.supported_backends))

        if not DataOrder.check_order_for_engine(backend, self.acquisition_geometry):
            raise ValueError("Input data must be reordered for use with selected backend. Use input.reorder('{0}')".format(backend))

        #set ProjectionOperator class from backend
        try:
            module = importlib.import_module("cil.plugins." + backend)
        except ImportError:
            if backend == "tigre":
                raise ImportError("Cannot import the {} plugin module. Please install TIGRE or select a different backend".format(self.backend))
            if backend == "astra":
                raise ImportError("Cannot import the {} plugin module. Please install CIL-ASTRA or select a different backend".format(self.backend))

        self._PO_class = module.ProjectionOperator
        self._backend = backend

    def _str_data_size(self) -> str:
        repres = "\nInput Data:\n"
        for dim in zip(self.acquisition_geometry.dimension_labels, self.acquisition_geometry.shape):
            repres += "\t" + str(dim[0]) + ": " + str(dim[1])+"\n"

        repres += "\nReconstruction Volume:\n"
        for dim in zip(self.image_geometry.dimension_labels, self.image_geometry.shape):
            repres += "\t" + str(dim[0]) + ": " + str(dim[1]) +"\n"

        return repres
