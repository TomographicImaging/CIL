#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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


from cil.framework import AcquisitionGeometry, Processor, ImageData
from cil.plugins.astra.processors.FDK_Flexible import FDK_Flexible
from cil.plugins.astra.utilities import convert_geometry_to_astra_vec_3D, convert_geometry_to_astra
import logging
import astra
import numpy as np

class FBP_Flexible(FDK_Flexible):

    """
    FBP_Flexible Filtered Back Projection performs an FBP reconstruction for 2D and 3D parallel-beam geometries.
    It is able to back-project circular trajectories with 2 PI angular range and equally spaced angular steps.

    This uses the ram-lak filter
    This is a GPU version only

    Parameters
    ----------
    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    sinogram_geometry : AcquisitionGeometry
        A description of the acquisition data

    Example
    -------
    >>> from cil.plugins.astra import FBP_Flexible
    >>> fbp = FBP_Flexible(image_geometry, data.geometry)
    >>> fbp.set_input(data)
    >>> reconstruction = fbp.get_ouput()

    Note
    ----
    ASTRA-toolbox provides limited FBP GPU support for offset/rotated geometries.
    This class creates a pseudo cone-beam geometry in-order to leverage the ASTRA FDK algorithm.
    """

    def __init__(self, volume_geometry,
                       sinogram_geometry):

        super(FBP_Flexible, self).__init__( volume_geometry = volume_geometry, sinogram_geometry = sinogram_geometry)

        #convert parallel geomerty to cone with large source to object
        sino_geom_cone = sinogram_geometry.copy()

        #this catches behaviour modified after CIL 21.3.1
        try:
            sino_geom_cone.config.system.align_reference_frame('cil')
        except:
            sino_geom_cone.config.system.update_reference_frame()

        #reverse ray direction unit-vector direction and extend to inf
        cone_source = -sino_geom_cone.config.system.ray.direction * sino_geom_cone.config.panel.pixel_size[1] * sino_geom_cone.config.panel.num_pixels[1] * 1e6
        detector_position = sino_geom_cone.config.system.detector.position
        detector_direction_x = sino_geom_cone.config.system.detector.direction_x

        if sino_geom_cone.dimension == '2D':
            tmp = AcquisitionGeometry.create_Cone2D(cone_source, detector_position, detector_direction_x)
        else:
            detector_direction_y = sino_geom_cone.config.system.detector.direction_y
            tmp = AcquisitionGeometry.create_Cone3D(cone_source, detector_position, detector_direction_x, detector_direction_y)

        sino_geom_cone.config.system = tmp.config.system.copy()

        self.vol_geom_astra, self.proj_geom_astra = convert_geometry_to_astra_vec_3D(volume_geometry, sino_geom_cone)

    def check_input(self, dataset):

        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))

        if self.sinogram_geometry.geom_type != 'parallel':
            raise ValueError("Expected input data to be parallel beam geometry , got {0}"\
                 .format(self.sinogram_geometry.geom_type))

        return True


class FBP_CPU(Processor):

    """
    FBP_CPU Filtered Back Projection performs an FBP reconstruction for 2D parallel-beam geometries.
    It is able to back-project circular trajectories with 2 PI angular range and equally spaced angular steps.

    This uses the ram-lak filter
    This is a CPU version only

    Parameters
    ----------
    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    sinogram_geometry : AcquisitionGeometry
        A description of the acquisition data

    Example
    -------
    >>> from cil.plugins.astra import FBP_CPU
    >>> fbp = FBP_CPU(image_geometry, data.geometry)
    >>> fbp.set_input(data)
    >>> reconstruction = fbp.get_ouput()
    """


    def __init__(self, volume_geometry,
                       sinogram_geometry):

        super().__init__( volume_geometry = volume_geometry, sinogram_geometry = sinogram_geometry)


    def check_input(self, dataset):

        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))

        if self.sinogram_geometry.geom_type != 'parallel':
            raise ValueError("Expected input data to be parallel beam geometry , got {0}"\
                 .format(self.sinogram_geometry.geom_type))

        if self.sinogram_geometry.dimension != '2D':
            raise ValueError("Expected input data to be 2D , got {0}"\
                 .format(self.sinogram_geometry.dimension))

        if self.sinogram_geometry.system_description != 'simple':
            logging.WARNING("The ASTRA backend FBP will use simple geometry only. Any configuration offsets or rotations may be ignored.")

        return True


    def process(self, out=None):

        # Get DATA
        DATA = self.get_input()

        vol_geom_astra, proj_geom_astra = convert_geometry_to_astra(self.volume_geometry, self.sinogram_geometry)

        rec_id = astra.data2d.create('-vol', vol_geom_astra)
        sinogram_id = astra.data2d.create('-sino', proj_geom_astra, DATA.as_array())
        cfg = astra.astra_dict('FBP')
        cfg['ReconstructionDataId'] = rec_id
        cfg['ProjectionDataId'] = sinogram_id
        cfg['ProjectorId'] = astra.create_projector('line', proj_geom_astra, vol_geom_astra)
        cfg['FilterType'] = 'ram-lak'

        alg_id = astra.algorithm.create(cfg)

        astra.algorithm.run(alg_id)
        arr_out = astra.data2d.get(rec_id)

        astra.data2d.delete(rec_id)
        astra.data2d.delete(sinogram_id)
        astra.algorithm.delete(alg_id)

        arr_out = np.flip(arr_out, 0)
        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)
