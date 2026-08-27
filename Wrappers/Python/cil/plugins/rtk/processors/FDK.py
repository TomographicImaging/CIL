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


from cil.framework import DataProcessor, ImageData
from cil.plugins.rtk.utilities import convert_geometry_to_rtk, make_constant_image_source, cil_acquisition_data_to_rtk_acquisition_data
import numpy as np
import itk
from itk import RTK as rtk

ImageType = itk.Image[itk.F, 3]

class FDK(DataProcessor):

    """
    FDK_Flexible Filtered Back Projection performs an FDK reconstruction for 2D and 3D cone-beam geometries.
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
    >>> from cil.plugins.rtk import FDK
    >>> fbp = FDK(image_geometry, data.geometry)
    >>> fbp.set_input(data)
    >>> reconstruction = fbp.get_ouput()

    """

    def __init__(self, volume_geometry,
                       sinogram_geometry):

        vol_geom_itk, proj_geom_itk = convert_geometry_to_rtk(volume_geometry, sinogram_geometry)

        super(FDK, self).__init__( volume_geometry = volume_geometry,
                                        sinogram_geometry = sinogram_geometry,
                                        vol_geom_itk = vol_geom_itk,
                                        proj_geom_itk = proj_geom_itk)



    def check_input(self, dataset):

        if self.sinogram_geometry.channels != 1:
            raise ValueError("Expected input data to be single channel, got {0}"\
                 .format(self.sinogram_geometry.channels))

        if self.sinogram_geometry.geom_type != 'cone' and self.sinogram_geometry.geom_type != 'cone_flex':
            raise ValueError("Expected input data to be cone beam geometry , got {0}"\
                 .format(self.sinogram_geometry.geom_type))

        return True
    
    def _set_up(self):
        """
        Configure processor attributes that require the data to setup
        Must set _shape_out
        """
        self._shape_out = self.volume_geometry.shape
    
    def process(self, out=None):

        # Get DATA
        DATA = self.get_input()

        vol_template = make_constant_image_source(self.vol_geom_itk, constant=0.0)

        sin_template = cil_acquisition_data_to_rtk_acquisition_data(DATA)

        FDKType = rtk.FDKConeBeamReconstructionFilter[ImageType]
        fdk = FDKType.New()
        fdk.SetInput(0, vol_template.GetOutput())
        fdk.SetInput(1, sin_template)
        fdk.SetGeometry(self.proj_geom_itk)

        # TODO, need??
        fdk.GetRampFilter().SetTruncationCorrection(0.0)
        fdk.GetRampFilter().SetHannCutFrequency(0.0)
        fdk.Update()
        recon_rtk = fdk.GetOutput() 

        arr_out = itk.GetArrayFromImage(recon_rtk)
              
        if out is None:
            out = ImageData(arr_out, deep_copy=False, geometry=self.volume_geometry.copy())
        else:
            out.fill(arr_out)
        return out
        

  