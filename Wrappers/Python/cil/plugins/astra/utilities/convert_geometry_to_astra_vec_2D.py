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


import astra
import numpy as np
from cil.framework.labels import AngleUnit
from cil.framework.labels import AcquisitionType

def convert_geometry_to_astra_vec_2D(volume_geometry, sinogram_geometry_in):

    """
    Converts CIL 2D geometries to ASTRA 2D vector Geometries.

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
    sinogram_geometry = sinogram_geometry_in.copy()

    if sinogram_geometry.geom_type == AcquisitionType.CONE_FLEX:
        raise ValueError('Cone-Flex geometry is not supported by this function, use convert_geometry_to_astra_vec_3D instead')

    sinogram_geometry.config.system.align_reference_frame('cil')
    angles = sinogram_geometry.config.angles
    system = sinogram_geometry.config.system
    panel = sinogram_geometry.config.panel

    #get units
    degrees = angles.angle_unit == AngleUnit.DEGREE

    #create a 2D astra geom from 2D CIL geometry, 2D astra geometry has axis flipped compared to 3D
    volume_geometry_temp = volume_geometry.copy()

    row = np.zeros((2,1))
    row[0] = panel.pixel_size[0] * system.detector.direction_x[0]
    row[1] = panel.pixel_size[0] * system.detector.direction_x[1]

    if 'right' in panel.origin:
        row *= -1

    det = np.zeros((2,1))
    det[0] = system.detector.position[0]
    det[1] = -system.detector.position[1]

    src = np.zeros((2,1))
    if sinogram_geometry.geom_type == 'parallel':
        src[0] = system.ray.direction[0]
        src[1] = -system.ray.direction[1]
        projector = 'parallel_vec'
    else:
        src[0] = system.source.position[0]
        src[1] = -system.source.position[1]
        projector = 'fanflat_vec'

    vectors = np.zeros((angles.num_positions, 6))

    for i, theta in enumerate(angles.angle_data):
        ang = + angles.initial_angle + theta

        rotation_matrix = rotation_matrix_z_from_euler(ang, degrees=degrees)

        vectors[i, :2]  = rotation_matrix.dot(src).reshape(2)
        vectors[i, 2:4] = rotation_matrix.dot(det).reshape(2)
        vectors[i, 4:6] = rotation_matrix.dot(row).reshape(2)

    proj_geom = astra.creators.create_proj_geom(projector, panel.num_pixels[0], vectors)
    vol_geom = astra.create_vol_geom(volume_geometry_temp.voxel_num_y,
                                    volume_geometry_temp.voxel_num_x,
                                    volume_geometry_temp.get_min_x(),
                                    volume_geometry_temp.get_max_x(),
                                    volume_geometry_temp.get_min_y(),
                                    volume_geometry_temp.get_max_y(),
                                    )

    return vol_geom, proj_geom

def rotation_matrix_z_from_euler(angle, degrees):
    """
    Returns 2D rotation matrix for z axis using direction cosine

    Parameters
    ----------
    angle : float
        angle or rotation around z axis

    degrees : bool
        if radian or degrees
    """
    if degrees:
        alpha = angle / 180. * np.pi
    else:
        alpha = angle

    rot_matrix = np.zeros((2,2), dtype=np.float64)
    rot_matrix[0][0] = np.cos(alpha)
    rot_matrix[0][1] = -np.sin(alpha)
    rot_matrix[1][0] = np.sin(alpha)
    rot_matrix[1][1] = np.cos(alpha)

    return rot_matrix
