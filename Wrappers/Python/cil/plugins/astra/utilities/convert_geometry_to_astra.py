#  Copyright 2018 United Kingdom Research and Innovation
#  Copyright 2018 The University of Manchester
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


import astra
import numpy as np
from cil.framework import acquisition_labels

def convert_geometry_to_astra(volume_geometry, sinogram_geometry):
    """
    Converts CIL geometries to simple ASTRA Geometries. Any offsets/rotations will be ignored.

    Parameters
    ----------
    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    sinogram_geometry : AcquisitionGeometry
        A description of the acquisition data

    Returns
    -------
    astra_volume_geom, astra_projection_geom
        The ASTRA vol_geom and proj_geom

    """

    # determine if the geometry is 2D or 3D

    if sinogram_geometry.pixel_num_v > 1:
        dimension = '3D'
    else:
        dimension = '2D'

    #get units

    if sinogram_geometry.config.angles.angle_unit == acquisition_labels["DEGREE"]:
        angles_rad = sinogram_geometry.config.angles.angle_data * np.pi / 180.0
    else:
        angles_rad = sinogram_geometry.config.angles.angle_data

    if dimension == '2D':
        vol_geom = astra.create_vol_geom(volume_geometry.voxel_num_y,
                                         volume_geometry.voxel_num_x,
                                         volume_geometry.get_min_x(),
                                         volume_geometry.get_max_x(),
                                         volume_geometry.get_min_y(),
                                         volume_geometry.get_max_y())

        if sinogram_geometry.geom_type == 'parallel':
            proj_geom = astra.create_proj_geom('parallel',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_num_h,
                                               -angles_rad)
        elif sinogram_geometry.geom_type == 'cone':
            proj_geom = astra.create_proj_geom('fanflat',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_num_h,
                                               -angles_rad,
                                               np.abs(sinogram_geometry.dist_source_center),
                                               np.abs(sinogram_geometry.dist_center_detector))
        else:
            NotImplemented

    elif dimension == '3D':
        vol_geom = astra.create_vol_geom(volume_geometry.voxel_num_y,
                                         volume_geometry.voxel_num_x,
                                         volume_geometry.voxel_num_z,
                                         volume_geometry.get_min_x(),
                                         volume_geometry.get_max_x(),
                                         volume_geometry.get_min_y(),
                                         volume_geometry.get_max_y(),
                                         volume_geometry.get_min_z(),
                                         volume_geometry.get_max_z())

        if sinogram_geometry.geom_type == 'parallel':
            proj_geom = astra.create_proj_geom('parallel3d',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_size_v,
                                               sinogram_geometry.pixel_num_v,
                                               sinogram_geometry.pixel_num_h,
                                               -angles_rad)
        elif sinogram_geometry.geom_type == 'cone':
            proj_geom = astra.create_proj_geom('cone',
                                               sinogram_geometry.pixel_size_h,
                                               sinogram_geometry.pixel_size_v,
                                               sinogram_geometry.pixel_num_v,
                                               sinogram_geometry.pixel_num_h,
                                               -angles_rad,
                                               np.abs(sinogram_geometry.dist_source_center),
                                               np.abs(sinogram_geometry.dist_center_detector))
        else:
            NotImplemented

    else:
        NotImplemented

    return vol_geom, proj_geom
