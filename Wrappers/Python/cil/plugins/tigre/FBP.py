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
import contextlib
import io

import numpy as np

from cil.framework import DataProcessor, ImageData
from cil.framework.labels import AcquisitionDimensionLabels, ImageDimensionLabels
from cil.plugins.tigre import CIL2TIGREGeometry

try:
    from tigre.algorithms import fdk, fbp
except ModuleNotFoundError:
    raise ModuleNotFoundError("This plugin requires the additional package TIGRE\n" +
            "Please install it via conda as tigre from the ccpi channel")

class FBP(DataProcessor):

    '''FBP Filtered Back Projection is a reconstructor for 2D and 3D parallel and cone-beam geometries.
    It is able to back-project circular trajectories with 2 PI angular range and equally spaced angular steps.

    This uses the ram-lak filter
    This is provided for simple and offset parallel-beam geometries only

    acquisition_geometry : AcquisitionGeometry
        A description of the acquisition data

    image_geometry : ImageGeometry, default used if None
        A description of the area/volume to reconstruct

    Example
    -------
    >>> from cil.plugins.tigre import FBP
    >>> fbp = FBP(image_geometry, data.geometry)
    >>> fbp.set_input(data)
    >>> reconstruction = fbp.get_output()

    '''

    def __init__(self, image_geometry=None, acquisition_geometry=None, **kwargs):
        if acquisition_geometry is None:
            raise TypeError("Please specify an acquisition_geometry to configure this processor")
        if image_geometry is None:
            image_geometry = acquisition_geometry.get_ImageGeometry()

        device = kwargs.get('device', 'gpu')
        if device != 'gpu':
            raise ValueError("TIGRE FBP is GPU only. Got device = {}".format(device))


        AcquisitionDimensionLabels.check_order_for_engine('tigre', acquisition_geometry)
        ImageDimensionLabels.check_order_for_engine('tigre', image_geometry)


        tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(image_geometry,acquisition_geometry)

        super(FBP, self).__init__(  image_geometry = image_geometry, acquisition_geometry = acquisition_geometry,\
                                    tigre_geom=tigre_geom, tigre_angles=tigre_angles)


    def check_input(self, dataset):

        if self.acquisition_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.acquisition_geometry.channels))

        AcquisitionDimensionLabels.check_order_for_engine('tigre', dataset.geometry)
        return True

    def process(self, out=None):

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(self.get_input().as_array(), axis=1)

            if self.acquisition_geometry.geom_type == 'cone':
                # suppress print statements from TIGRE https://github.com/CERN/TIGRE/issues/532
                with contextlib.redirect_stdout(io.StringIO()):
                    arr_out = fdk(data_temp, self.tigre_geom, self.tigre_angles)
            else:
                arr_out = fbp(data_temp, self.tigre_geom, self.tigre_angles)
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            if self.acquisition_geometry.geom_type == 'cone':
                # suppress print statements from TIGRE https://github.com/CERN/TIGRE/issues/532
                with contextlib.redirect_stdout(io.StringIO()):
                    arr_out = fdk(self.get_input().as_array(), self.tigre_geom, self.tigre_angles)
            else:
                arr_out = fbp(self.get_input().as_array(), self.tigre_geom, self.tigre_angles)

        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.image_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
