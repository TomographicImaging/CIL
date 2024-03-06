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

from cil.framework import DataProcessor
from cil.framework import DataOrder
from cil.plugins.astra.processors.FBP_Flexible import FBP_Flexible
from cil.plugins.astra.processors.FDK_Flexible import FDK_Flexible
from cil.plugins.astra.processors.FBP_Flexible import FBP_CPU
import logging

class FBP(DataProcessor):

    """
    FBP configures and calls an appropriate ASTRA FBP or FDK algorithm for your dataset.

    The best results will be on data with circular trajectories of a 2PI angular range and equally spaced small angular steps.

    Parameters
    ----------
    image_geometry : ImageGeometry, default used if None
        A description of the area/volume to reconstruct

    acquisition_geometry : AcquisitionGeometry
        A description of the acquisition data

    device : string, default='gpu'
        'gpu' will run on a compatible CUDA capable device using the ASTRA FDK_CUDA algorithm
        'cpu' will run on CPU using the ASTRA FBP algorithm - see Notes for limitations


    Example
    -------
    >>> from cil.plugins.astra import FBP
    >>> fbp = FBP(image_geometry, data.geometry)
    >>> fbp.set_input(data)
    >>> reconstruction = fbp.get_ouput()


    Notes
    -----
    A CPU version is provided for simple 2D parallel-beam geometries only, any offsets and rotations in the acquisition geometry will be ignored.

    This uses the ram-lak filter only.

    """


    def __init__(self, image_geometry=None, acquisition_geometry=None, device='gpu', **kwargs):


        sinogram_geometry = kwargs.get('sinogram_geometry', None)
        volume_geometry = kwargs.get('volume_geometry', None)

        if sinogram_geometry is not None:
            acquisition_geometry = sinogram_geometry
            logging.warning("sinogram_geometry has been deprecated. Please use acquisition_geometry instead.")

        if acquisition_geometry is None:
            raise TypeError("Please specify an acquisition_geometry to configure this processor")

        if volume_geometry is not None:
            image_geometry = volume_geometry
            logging.warning("volume_geometry has been deprecated. Please use image_geometry instead.")

        if image_geometry is None:
            image_geometry = acquisition_geometry.get_ImageGeometry()

        DataOrder.check_order_for_engine('astra', image_geometry)
        DataOrder.check_order_for_engine('astra', acquisition_geometry)

        if device == 'gpu':
            if acquisition_geometry.geom_type == 'parallel':
                processor = FBP_Flexible(image_geometry, acquisition_geometry)
            else:
                processor = FDK_Flexible(image_geometry, acquisition_geometry)

        else:
            UserWarning("ASTRA back-projector running on CPU will not make use of enhanced geometry parameters")

            if acquisition_geometry.geom_type == 'cone':
                raise NotImplementedError("Cannot process cone-beam data without a GPU")

            if acquisition_geometry.dimension == '2D':
                processor = FBP_CPU(image_geometry, acquisition_geometry)
            else:
                raise NotImplementedError("Cannot process 3D data without a GPU")

        if acquisition_geometry.channels > 1:
            raise NotImplementedError("Cannot process multi-channel data")
            #processor_full = ChannelwiseProcessor(processor, self.acquisition_geometry.channels, dimension='prepend')
            #self.processor = operator_full

        super(FBP, self).__init__( image_geometry=image_geometry, acquisition_geometry=acquisition_geometry, device=device, processor=processor)

    def set_input(self, dataset):
        return self.processor.set_input(dataset)

    def get_input(self):
        return self.processor.get_input()

    def get_output(self, out=None):
        return self.processor.get_output(out=out)

    def check_input(self, dataset):
        return self.processor.check_input(dataset)

    def process(self, out=None):
        return self.processor.process(out=out)
