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


import numpy as np
from cil.framework.labels import AcquisitionType, AngleUnit
import itk
from itk import RTK as rtk

ImageType = itk.Image[itk.F, 3]

ImageType = itk.Image[itk.F, 3]

def make_constant_image_source(vol_geom, constant=0.0):
    """
    Return an RTK ConstantImageSource producing a 3D image.
    """

    size = vol_geom['size']
    spacing = vol_geom['spacing']
    origin = vol_geom['origin']
        
    src = rtk.ConstantImageSource[ImageType].New()
    src.SetSize([int(size[2]), int(size[1]), int(size[0])])
    src.SetSpacing([float(spacing[2]), float(spacing[1]), float(spacing[0])])
    src.SetOrigin([float(origin[2]), float(origin[1]), float(origin[0])])
    src.SetConstant(float(constant))
    return src

def convert_geometry_to_rtk(volume_geometry, sinogram_geometry):
    """
    Converts CIL geometries to RTK Geometries.

    Parameters
    ----------
    volume_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    sinogram_geometry : AcquisitionGeometry
        A description of the acquisition data

    Returns
    -------
    rtk_volume_geom, rtk_projection_geom
        The RTK vol_geom and proj_geom

    """

    # determine if the geometry is 2D or 3D
    dimension = AcquisitionType.DIM3 if sinogram_geometry.pixel_num_v > 1 else AcquisitionType.DIM2

    #get units

    if AcquisitionType.DIM2 & dimension:
        raise NotImplementedError("2D geometry is not implemented.") 

    elif AcquisitionType.DIM3 & dimension:

        vol_size = list(volume_geometry.shape)
        
        vol_spacing = (
            float(volume_geometry.voxel_size_z),
            float(volume_geometry.voxel_size_y),
            float(volume_geometry.voxel_size_x),
        )
    
        vol_origin = [ -(vol_size[0] - 1) * vol_spacing[0] / 2., 
                            -(vol_size[1] - 1) * vol_spacing[1] / 2., 
                            -(vol_size[2] - 1) * vol_spacing[2] / 2.]        

        if sinogram_geometry.geom_type == 'parallel':
            raise NotImplementedError("Parallel 3D geometry is not implemented.") 
            
        elif sinogram_geometry.geom_type == 'cone':

            src = np.asarray(sinogram_geometry.config.system.source.position, dtype=float)      
            det = np.asarray(sinogram_geometry.config.system.detector.position, dtype=float)    
            ang_rad = np.asarray(sinogram_geometry.angles, dtype=float)
        
            sid = float(np.linalg.norm(src - np.array([0.0, 0.0, 0.0])))
            sdd = float(np.linalg.norm(det - src))
        
            proj_geom = rtk.ThreeDCircularProjectionGeometry.New()
            for a in np.rad2deg(ang_rad):  
                proj_geom.AddProjection(sid, sdd, float(a))

        else:
            NotImplemented

    else:
        NotImplemented

    vol_geom = {'size':vol_size, 'spacing':vol_spacing, 'origin':vol_origin}
    
    return vol_geom, proj_geom


def cil_acquisition_data_to_rtk_acquisition_data(cil_data):
    """
    Convert CIL AcquisitionData (angle, vertical, horizontal) -> ITK image (u, v, angle) 
    """
    ag = cil_data.geometry

    ##TODO need to flip why?
    projs = np.asarray(cil_data.array[::-1], dtype=np.float32)

    proj_itk = itk.image_from_array(projs)

    du, dv = map(float, ag.config.panel.pixel_size)
    det_u, det_v = map(int, ag.config.panel.num_pixels)

    proj_itk.SetSpacing([du, dv, 1.0])

    proj_itk.SetOrigin([
        -(det_u - 1) * du / 2.0,
        -(det_v - 1) * dv / 2.0,
        0.0
    ])

    return proj_itk