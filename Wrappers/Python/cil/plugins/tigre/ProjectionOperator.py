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

from cil.framework import ImageData, AcquisitionData, AcquisitionGeometry
from cil.framework import DataOrder
from cil.framework.BlockGeometry import BlockGeometry
from cil.optimisation.operators import BlockOperator
from cil.optimisation.operators import LinearOperator
from cil.plugins.tigre import CIL2TIGREGeometry
import numpy as np
import logging

try:
    from _Atb import _Atb_ext as Atb
    from _Ax import _Ax_ext as Ax

except ModuleNotFoundError:
    raise ModuleNotFoundError(
        "This plugin requires the additional package TIGRE\n" +
        "Please install it via conda as tigre from the ccpi channel")

try:
    from tigre.utilities.gpu import GpuIds
    has_gpu_sel = True
except ModuleNotFoundError:
    has_gpu_sel = False


class ProjectionOperator(LinearOperator):
    """
        ProjectionOperator configures and calls TIGRE Projectors for your dataset.

        Please refer to the TIGRE documentation for futher descriptions
        https://github.com/CERN/TIGRE
        https://iopscience.iop.org/article/10.1088/2057-1976/2/5/055010


        Parameters
        ----------

        image_geometry : `ImageGeometry`, default used if None
            A description of the area/volume to reconstruct

        acquisition_geometry :`AcquisitionGeometry`, `BlockGeometry`
            A description of the acquisition data. If passed a BlockGeometry it will return a BlockOperator.

        direct_method : str,  default 'interpolated'
            The method used by the forward projector, 'Siddon' for ray-voxel intersection, 'interpolated' for interpolated projection

        adjoint_weights : str, default 'matched'
            The weighting method used by the cone-beam backward projector, 'matched' for weights to approximately match the 'interpolated' forward projector, 'FDK' for FDK weights

        Example
        -------
        >>> from cil.plugins.tigre import ProjectionOperator
        >>> PO = ProjectionOperator(image.geometry, data.geometry)
        >>> forward_projection = PO.direct(image)
        >>> backward_projection = PO.adjoint(data)

        """
    def __new__(cls, image_geometry=None, acquisition_geometry=None, \
        direct_method='interpolated',adjoint_weights='matched', **kwargs):
        if isinstance(acquisition_geometry, BlockGeometry):
            log.info("BlockOperator is returned.")

            K = []
            for ag in acquisition_geometry:
                K.append(
                    ProjectionOperator_ag(image_geometry=image_geometry, acquisition_geometry=ag, \
                        direct_method=direct_method, adjoint_weights=adjoint_weights, **kwargs)
                )
            return BlockOperator(*K)
        else:
            log.info("Standard Operator is returned.")
            return super(ProjectionOperator,
                         cls).__new__(ProjectionOperator_ag)


class ProjectionOperator_ag(ProjectionOperator):
    '''TIGRE Projection Operator'''

    def __init__(self,
                 image_geometry=None,
                 acquisition_geometry=None,
                 direct_method='interpolated',
                 adjoint_weights='matched',
                 **kwargs):
        """
        ProjectionOperator configures and calls TIGRE Projectors for your dataset.

        Please refer to the TIGRE documentation for futher descriptions
        https://github.com/CERN/TIGRE
        https://iopscience.iop.org/article/10.1088/2057-1976/2/5/055010


        Parameters
        ----------

        image_geometry : ImageGeometry, default used if None
            A description of the area/volume to reconstruct

        acquisition_geometry : AcquisitionGeometry
            A description of the acquisition data

        direct_method : str,  default 'interpolated'
            The method used by the forward projector, 'Siddon' for ray-voxel intersection, 'interpolated' for interpolated projection

        adjoint_weights : str, default 'matched'
            The weighting method used by the cone-beam backward projector, 'matched' for weights to approximately match the 'interpolated' forward projector, 'FDK' for FDK weights

        Example
        -------
        >>> from cil.plugins.tigre import ProjectionOperator
        >>> PO = ProjectionOperator(image.geometry, data.geometry)
        >>> forward_projection = PO.direct(image)
        >>> backward_projection = PO.adjoint(data)

        """

        acquisition_geometry_old = kwargs.get('aquisition_geometry', None)

        if acquisition_geometry_old is not None:
            acquisition_geometry = acquisition_geometry_old
            logging.warning(
                "aquisition_geometry has been deprecated. Please use acquisition_geometry instead."
            )

        if acquisition_geometry is None:
            raise TypeError(
                "Please specify an acquisition_geometry to configure this operator"
            )

        if image_geometry == None:
            image_geometry = acquisition_geometry.get_ImageGeometry()

        device = kwargs.get('device', 'gpu')
        if device != 'gpu':
            raise ValueError(
                "TIGRE projectors are GPU only. Got device = {}".format(
                    device))

        DataOrder.check_order_for_engine('tigre', image_geometry)
        DataOrder.check_order_for_engine('tigre', acquisition_geometry)

        super(ProjectionOperator,self).__init__(domain_geometry=image_geometry,\
             range_geometry=acquisition_geometry)

        if direct_method not in ['interpolated', 'Siddon']:
            raise ValueError(
                "direct_method expected 'interpolated' or 'Siddon' got {}".
                format(direct_method))

        if adjoint_weights not in ['matched', 'FDK']:
            raise ValueError(
                "adjoint_weights expected 'matched' or 'FDK' got {}".format(
                    adjoint_weights))

        self.method = {'direct': direct_method, 'adjoint': adjoint_weights}

        #set up TIGRE geometry
        tigre_geom, tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            image_geometry, acquisition_geometry)

        tigre_geom.check_geo(tigre_angles)
        tigre_geom.cast_to_single()
        self.tigre_geom = tigre_geom

        #set up TIGRE GPU targets (from 2.2)
        if has_gpu_sel:
            self.gpuids = GpuIds()

    def __call_Ax(self, data):
        if has_gpu_sel:
            return Ax(data, self.tigre_geom, self.tigre_geom.angles,
                      self.method['direct'], self.tigre_geom.mode, self.gpuids)
        else:
            return Ax(data, self.tigre_geom, self.tigre_geom.angles,
                      self.method['direct'], self.tigre_geom.mode)

    def direct(self, x, out=None):

        data = x.as_array()

        if self.tigre_geom.is2D:
            data_temp = np.expand_dims(data, axis=0)
            arr_out = self.__call_Ax(data_temp)
            arr_out = np.squeeze(arr_out, axis=1)
        else:
            arr_out = self.__call_Ax(data)

        #if single angle projection remove the dimension for CIL
        if arr_out.shape[0] == 1:
            arr_out = np.squeeze(arr_out, axis=0)

        if out is None:
            out = AcquisitionData(arr_out,
                                  deep_copy=False,
                                  geometry=self._range_geometry.copy(),
                                  suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def __call_Atb(self, data):
        if has_gpu_sel:
            return Atb(data, self.tigre_geom, self.tigre_geom.angles,
                       self.method['adjoint'], self.tigre_geom.mode,
                       self.gpuids)
        else:
            return Atb(data, self.tigre_geom, self.tigre_geom.angles,
                       self.method['adjoint'], self.tigre_geom.mode)

    def adjoint(self, x, out=None):

        data = x.as_array()

        #if single angle projection add the dimension in for TIGRE
        if x.dimension_labels[0] != AcquisitionGeometry.ANGLE:
            data = np.expand_dims(data, axis=0)

        if self.tigre_geom.is2D:
            data = np.expand_dims(data, axis=1)
            arr_out = self.__call_Atb(data)
            arr_out = np.squeeze(arr_out, axis=0)
        else:
            arr_out = self.__call_Atb(data)

        if out is None:
            out = ImageData(arr_out,
                            deep_copy=False,
                            geometry=self._domain_geometry.copy(),
                            suppress_warning=True)
            return out
        else:
            out.fill(arr_out)

    def domain_geometry(self):
        return self._domain_geometry

    def range_geometry(self):
        return self._range_geometry
