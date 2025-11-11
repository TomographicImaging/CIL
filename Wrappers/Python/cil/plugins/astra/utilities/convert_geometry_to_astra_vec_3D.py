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
from cil.framework.labels import AcquisitionType, AngleUnit

def convert_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in):
    """
    Converts CIL 2D and 3D geometries to ASTRA 3D vector Geometries.

    Selects the appropriate conversion function based on the type of CIL acquisition geometry.

    Parameters
    ----------
    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct
    sinogram_geometry_in : AcquisitionGeometry
        A description of the acquisition data

    Returns
    -------
    astra_volume_geom, astra_projection_geom
        The ASTRA vol_geom and proj_geom
    """

    if sinogram_geometry_in.geom_type == AcquisitionType.CONE_FLEX:
        return convert_flex_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in)
    
    return convert_standard_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in)


def convert_standard_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in):
    """
    Converts CIL 2D and 3D circular geometries to ASTRA 3D vector Geometries.

    Parameters
    ----------
    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct
    sinogram_geometry_in : AcquisitionGeometry
        A description of the acquisition data

    Returns
    -------
    astra_volume_geom, astra_projection_geom
        The ASTRA vol_geom and proj_geom

    """
    if sinogram_geometry_in.geom_type not in [AcquisitionType.PARALLEL, AcquisitionType.CONE]:
        raise ValueError(f"Unsupported geometry type: {sinogram_geometry_in.geom_type}. Only {AcquisitionType.PARALLEL} and {AcquisitionType.CONE}  geometries are supported.")
    
    sinogram_geometry = sinogram_geometry_in.copy()
    volume_geometry_temp = volume_geometry.copy()

    sinogram_geometry.config.system.align_reference_frame('cil')
    angles = sinogram_geometry.config.angles
    degrees = angles.angle_unit == AngleUnit.DEGREE

    system = sinogram_geometry.config.system
    panel = sinogram_geometry.config.panel
    

    if AcquisitionType.DIM2 & sinogram_geometry.dimension:
        #create a 3D astra geom from 2D CIL geometry
        volume_geometry_temp.voxel_num_z = 1

        volume_geometry_temp.voxel_size_z = volume_geometry_temp.voxel_size_x
        panel.pixel_size[1] =  volume_geometry_temp.voxel_size_z * sinogram_geometry.magnification

        row = np.zeros((3,1))
        row[0] = panel.pixel_size[0] * system.detector.direction_x[0]
        row[1] = panel.pixel_size[0] * system.detector.direction_x[1]

        if 'right' in panel.origin:
            row *= -1

        col = np.zeros((3,1))
        col[2] = panel.pixel_size[1]

        det = np.zeros((3,1))
        det[0] = system.detector.position[0]
        det[1] = system.detector.position[1]

        src = np.zeros((3,1))
        if sinogram_geometry.geom_type == 'parallel':
            src[0] = system.ray.direction[0]
            src[1] = system.ray.direction[1]
            projector = 'parallel3d_vec'
        else:
            src[0] = system.source.position[0]
            src[1] = system.source.position[1]
            projector = 'cone_vec'

    else:
        row = panel.pixel_size[0] * system.detector.direction_x.reshape(3,1)
        col = panel.pixel_size[1] * system.detector.direction_y.reshape(3,1)
        det = system.detector.position.reshape(3, 1)

        if 'right' in panel.origin:
            row *= -1
        if 'top' in panel.origin:
            col *= -1

        if sinogram_geometry.geom_type == 'parallel':
            src = system.ray.direction.reshape(3,1)
            projector = 'parallel3d_vec'
        elif sinogram_geometry.geom_type != 'cone_flex':
            src = system.source.position.reshape(3,1)
            projector = 'cone_vec'
  

    #Build for astra 3D only
    vectors = np.zeros((angles.num_positions, 12))

    for i, theta in enumerate(angles.angle_data):
        ang = - angles.initial_angle - theta

        rotation_matrix = rotation_matrix_z_from_euler(ang, degrees=degrees)

        vectors[i, :3]  = rotation_matrix.dot(src).reshape(3)
        vectors[i, 3:6] = rotation_matrix.dot(det).reshape(3)
        vectors[i, 6:9] = rotation_matrix.dot(row).reshape(3)
        vectors[i, 9:]  = rotation_matrix.dot(col).reshape(3)    

    proj_geom = astra.creators.create_proj_geom(projector, panel.num_pixels[1], panel.num_pixels[0], vectors)
    vol_geom = astra.create_vol_geom(volume_geometry_temp.voxel_num_y,
                                    volume_geometry_temp.voxel_num_x,
                                    volume_geometry_temp.voxel_num_z,
                                    volume_geometry_temp.get_min_x(),
                                    volume_geometry_temp.get_max_x(),
                                    volume_geometry_temp.get_min_y(),
                                    volume_geometry_temp.get_max_y(),
                                    volume_geometry_temp.get_min_z(),
                                    volume_geometry_temp.get_max_z()
                                    )


    return vol_geom, proj_geom


def convert_flex_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in):
    """
    Converts CIL CONE_FLEX geometries to ASTRA 3D vector Geometries.

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

    if sinogram_geometry_in.geom_type != AcquisitionType.CONE_FLEX:
        raise ValueError(f"Unsupported geometry type: {sinogram_geometry_in.geom_type}. Only {AcquisitionType.CONE_FLEX} geometries are supported.")
    

    system = sinogram_geometry_in.config.system
    panel = sinogram_geometry_in.config.panel
     
    sign_h = 1
    sign_v = 1

    if 'right' in panel.origin:
        sign_h = -1
    if 'top' in panel.origin:
        sign_v = -1

    projector = 'cone_vec'
    vectors = np.zeros((system.num_positions, 12))

    for i in range(system.num_positions):
        vectors[i, :3] = system.source[i].position
        vectors[i, 3:6]  = system.detector[i].position
        vectors[i, 6:9] = sign_h * system.detector[i].direction_x * panel.pixel_size[0]
        vectors[i, 9:]  = sign_v * system.detector[i].direction_y * panel.pixel_size[1]

    proj_geom = astra.creators.create_proj_geom(projector, panel.num_pixels[1], panel.num_pixels[0], vectors)
    vol_geom = astra.create_vol_geom(volume_geometry.voxel_num_y,
                                    volume_geometry.voxel_num_x,
                                    volume_geometry.voxel_num_z,
                                    volume_geometry.get_min_x() + system.volume_centre.position[0],
                                    volume_geometry.get_max_x() + system.volume_centre.position[0],
                                    volume_geometry.get_min_y() + system.volume_centre.position[1],
                                    volume_geometry.get_max_y() + system.volume_centre.position[1],
                                    volume_geometry.get_min_z() + system.volume_centre.position[2],
                                    volume_geometry.get_max_z() + system.volume_centre.position[2]
                                    )


    return vol_geom, proj_geom

def rotation_matrix_z_from_euler(angle, degrees):

    """
    Returns 3D rotation matrix for z axis using direction cosine

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

    rot_matrix = np.zeros((3,3), dtype=np.float64)
    rot_matrix[0][0] = np.cos(alpha)
    rot_matrix[0][1] = - np.sin(alpha)
    rot_matrix[1][0] = np.sin(alpha)
    rot_matrix[1][1] = np.cos(alpha)
    rot_matrix[2][2] = 1

    return rot_matrix
