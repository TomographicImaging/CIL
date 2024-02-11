# -*- coding: utf-8 -*-
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


from typing import Literal, Optional, Union

import astra
import numpy as np
from astra import algorithm, astra_dict, data3d

from cil.framework import AcquisitionData, DataOrder, DataProcessor
from cil.framework.framework import (
    AcquisitionGeometry,
    ImageData,
    ImageGeometry,
)

from ..utilities import convert_geometry_to_astra_vec_3D


class AstraForwardProjector3D(DataProcessor):
    """AstraForwardProjector3D configures an ASTRA 3D forward projector for GPU.

    Parameters
    ----------
    volume_geometry
        A description of the area/volume to reconstruct

    sinogram_geometry
        A description of the acquisition data
    """

    def __init__(self, volume_geometry:Optional[ImageGeometry]=None, sinogram_geometry:Optional[AcquisitionGeometry]=None):
        kwargs = {
                  'volume_geometry'  : volume_geometry,
                  'sinogram_geometry'  : sinogram_geometry,
                  'proj_geom'  : None,
                  'vol_geom'  : None,
                  }

        super(AstraForwardProjector3D, self).__init__(**kwargs)

        self.set_ImageGeometry(volume_geometry)
        self.set_AcquisitionGeometry(sinogram_geometry)

        self.vol_geom, self.proj_geom = convert_geometry_to_astra_vec_3D(self.volume_geometry, self.sinogram_geometry)

    def check_input(self, dataset:ImageData) -> Literal[True]:
        """Check the dimension comparability of the passed dataset.

        Parameters
        ----------
        dataset
            Dataset to be checked

        Returns
        -------
        bool
            True if input checks are passed

        Raises
        ------
        ValueError
            Raised if input fails the dimension check
        """
        if self.volume_geometry.shape != dataset.geometry.shape:
            raise ValueError("Dataset not compatible with geometry used to create the projector")

        return True

    def set_ImageGeometry(self, volume_geometry:ImageGeometry) -> None:
        """Set the area/volume geometry to reconstruct.

        Parameters
        ----------
        volume_geometry
            Area/volume geometry
        """
        DataOrder.check_order_for_engine('astra', volume_geometry)

        if len(volume_geometry.dimension_labels) > 3:
            raise ValueError("Supports 2D and 3D data only, got {0}".format(volume_geometry.number_of_dimensions))

        self.volume_geometry = volume_geometry.copy()

    def set_AcquisitionGeometry(self, sinogram_geometry:AcquisitionGeometry) -> None:
        """Set the acquisition geometry of the data.

        Parameters
        ----------
        sinogram_geometry
            Acquisition geometry of the reconstructed data.
        """
        DataOrder.check_order_for_engine('astra', sinogram_geometry)

        if len(sinogram_geometry.dimension_labels) > 3:
            raise ValueError("Supports 2D and 3D data only, got {0}".format(sinogram_geometry.number_of_dimensions))

        self.sinogram_geometry = sinogram_geometry.copy()


    def process(self, out:Optional[ImageData]=None) -> Optional[AcquisitionData]:
        """Reconstruct the volume using ASTRA.

        Parameters
        ----------
        out
           Fills the referenced ImageData with the processed data and suppresses the return

        Returns
        -------
        Optional[AcquisitionData]
            Reconstructed data
        """
        IM = self.get_input()

        #ASTRA expects a 3D array with shape 1, CIL removes dimensions of len 1
        new_shape_ig = [self.volume_geometry.voxel_num_z,self.volume_geometry.voxel_num_y,self.volume_geometry.voxel_num_x]
        new_shape_ig = [x if x>0 else 1 for x in new_shape_ig]

        data_temp = IM.as_array().reshape(new_shape_ig)

        if out is None:
            sinogram_id, arr_out = astra.create_sino3d_gpu(data_temp,
                                                           self.proj_geom,
                                                           self.vol_geom)
        else:
            new_shape_ag = [self.sinogram_geometry.pixel_num_v,self.sinogram_geometry.num_projections,self.sinogram_geometry.pixel_num_h]
            arr_out = out.as_array().reshape(new_shape_ag)

            sinogram_id = astra.data3d.link('-sino', self.proj_geom, arr_out)
            self.create_sino3d_gpu(data_temp, self.proj_geom, self.vol_geom, False, sino_id=sinogram_id)

        #clear the memory on GPU
        astra.data3d.delete(sinogram_id)

        arr_out = np.squeeze(arr_out)

        if out is None:
            out = AcquisitionData(arr_out, deep_copy=False, geometry=self.sinogram_geometry.copy(), suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def create_sino3d_gpu(self, data:Union[np.ndarray, int], proj_geom:dict, vol_geom:dict, returnData:bool=True, gpuIndex:Optional[int]=None, sino_id:Optional[int]=None) -> Union[int, Union[int, np.ndarray]]:
        """Call to ASTRA to create a forward projection of an image (3D).

        Parameters
        ----------
        data
            Image data or ID.

        proj_geom
            Projection geometry.

        vol_geom
            Volume geometry.

        returnData
            If False, only return the ID of the forward projection.

        gpuIndex
            Optional GPU index.

        sino_id
            ID of the np array linked with astra.

        Returns
        -------
        proj_geom : int or (int, numpy.ndarray)
            If ``returnData=False``, returns the ID of the forward projection. Otherwise returns a tuple containing the ID of the forward projection and the forward projection itself.

        """
        if isinstance(data, np.ndarray):
            volume_id = data3d.create('-vol', vol_geom, data)
        else:
            volume_id = data
        if sino_id is None:
            sino_id = data3d.create('-sino', proj_geom, 0)

        algString = 'FP3D_CUDA'
        cfg = astra_dict(algString)
        if not gpuIndex==None:
            cfg['option']={'GPUindex':gpuIndex}
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = volume_id
        alg_id = algorithm.create(cfg)
        algorithm.run(alg_id)
        algorithm.delete(alg_id)

        if isinstance(data, np.ndarray):
            data3d.delete(volume_id)
        if sino_id is None:
            if returnData:
                return sino_id, data3d.get(sino_id)
            else:
                return sino_id
        else:
            if returnData:
                return sino_id, data3d.get_shared(sino_id)
            else:
                return sino_id
