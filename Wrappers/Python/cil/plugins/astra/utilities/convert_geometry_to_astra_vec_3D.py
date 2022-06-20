# -*- coding: utf-8 -*-
#  Copyright 2018 - 2022 United Kingdom Research and Innovation
#  Copyright 2018 - 2022 The University of Manchester
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


import astra
import numpy as np

def convert_geometry_to_astra_vec_3D(volume_geometry, sinogram_geometry_in):

    '''Set up ASTRA Volume and projection geometry, not stored

       :param volume_geometry: ccpi.framework.ImageGeometry
       :param sinogram_geometry: ccpi.framework.AcquisitionGeometry
       
       :returns ASTRA volume and sinogram geometry'''
 
    sinogram_geometry = sinogram_geometry_in.copy()
    
    #this catches behaviour modified after CIL 21.3.1 
    try:
        sinogram_geometry.config.system.align_reference_frame('cil')
    except:
        sinogram_geometry.config.system.update_reference_frame()
        

    angles = sinogram_geometry.config.angles
    system = sinogram_geometry.config.system
    panel = sinogram_geometry.config.panel

    #get units
    degrees = angles.angle_unit == sinogram_geometry.DEGREE
    
    if sinogram_geometry.dimension == '2D':
        #create a 3D astra geom from 2D CIL geometry
        volume_geometry_temp = volume_geometry.copy()
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
        volume_geometry_temp = volume_geometry.copy()
 
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
        else:
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

def rotation_matrix_z_from_euler(angle, degrees):
    '''Returns 3D rotation matrix for z axis using direction cosine

    :param angle: angle or rotation around z axis
    :type angle: float
    :param degrees: if radian or degrees
    :type bool: defines the unit measure of the angle
    '''
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
