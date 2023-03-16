# -*- coding: utf-8 -*-
#  Copyright 2022 United Kingdom Research and Innovation
#  Copyright 2022 The University of Manchester
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

from cil.framework import DataOrder
from cil.optimisation.operators import LinearOperator, ChannelwiseOperator
from cil.framework.BlockGeometry import BlockGeometry
from cil.optimisation.operators import BlockOperator
from cil.plugins.astra.operators import AstraProjector3D
from cil.plugins.astra.operators import AstraProjector2D
import logging


class ProjectionOperator(LinearOperator):
    """
    ProjectionOperator configures and calls appropriate ASTRA Projectors for your dataset.

    Parameters
    ----------

    image_geometry : ``ImageGeometry``, default used if None
        A description of the area/volume to reconstruct

    acquisition_geometry : ``AcquisitionGeometry``, ``BlockGeometry``
        A description of the acquisition data. If passed a BlockGeometry it will return a BlockOperator.

    device : string, default='gpu'
        'gpu' will run on a compatible CUDA capable device using the ASTRA 3D CUDA Projectors, 'cpu' will run on CPU using the ASTRA 2D CPU Projectors

    Example
    -------
    >>> from cil.plugins.astra import ProjectionOperator
    >>> PO = ProjectionOperator(image.geometry, data.geometry)
    >>> forward_projection = PO.direct(image)
    >>> backward_projection = PO.adjoint(data)

    Notes
    -----
    For multichannel data the ProjectionOperator will broadcast across all channels.
    """
    def __new__(cls, image_geometry=None, acquisition_geometry=None, \
        device='gpu', **kwargs):
        if isinstance(acquisition_geometry, BlockGeometry):
            logging.info("BlockOperator is returned.")

            K = []
            for ag in acquisition_geometry:
                K.append(
                    ProjectionOperator_ag(image_geometry=image_geometry, acquisition_geometry=ag, \
                        device=device, **kwargs)
                )
            return BlockOperator(*K)
        else:
            logging.info("Standard Operator is returned.")
            return super(ProjectionOperator,
                         cls).__new__(ProjectionOperator_ag)


class ProjectionOperator_ag(ProjectionOperator):
    """
    ProjectionOperator configures and calls appropriate ASTRA Projectors for your dataset.

    Parameters
    ----------

    image_geometry : ImageGeometry, default used if None
        A description of the area/volume to reconstruct

    acquisition_geometry : AcquisitionGeometry
        A description of the acquisition data

    device : string, default='gpu'
        'gpu' will run on a compatible CUDA capable device using the ASTRA 3D CUDA Projectors, 'cpu' will run on CPU using the ASTRA 2D CPU Projectors

    Example
    -------
    >>> from cil.plugins.astra import ProjectionOperator
    >>> PO = ProjectionOperator(image.geometry, data.geometry)
    >>> forward_projection = PO.direct(image)
    >>> backward_projection = PO.adjoint(data)

    Notes
    -----
    For multichannel data the ProjectionOperator will broadcast across all channels.
    """

    def __init__(self,
                 image_geometry=None,
                 acquisition_geometry=None,
                 device='gpu'):

        if acquisition_geometry is None:
            raise TypeError(
                "Please specify an acquisition_geometry to configure this operator"
            )

        if image_geometry is None:
            image_geometry = acquisition_geometry.get_ImageGeometry()

        super(ProjectionOperator_ag,
              self).__init__(domain_geometry=image_geometry,
                             range_geometry=acquisition_geometry)

        DataOrder.check_order_for_engine('astra', image_geometry)
        DataOrder.check_order_for_engine('astra', acquisition_geometry)

        self.volume_geometry = image_geometry
        self.sinogram_geometry = acquisition_geometry

        sinogram_geometry_sc = acquisition_geometry.get_slice(channel=0)
        volume_geometry_sc = image_geometry.get_slice(channel=0)

        if device == 'gpu':
            operator = AstraProjector3D(volume_geometry_sc,
                                        sinogram_geometry_sc)
        elif self.sinogram_geometry.dimension == '2D':
            operator = AstraProjector2D(volume_geometry_sc,
                                        sinogram_geometry_sc,
                                        device=device)
        else:
            raise NotImplementedError("Cannot process 3D data without a GPU")

        if acquisition_geometry.channels > 1:
            operator_full = ChannelwiseOperator(
                operator, self.sinogram_geometry.channels, dimension='prepend')
            self.operator = operator_full
        else:
            self.operator = operator

    def direct(self, IM, out=None):
        '''Applies the direct of the operator i.e. the forward projection.
        
        Parameters
        ----------
        IM : ImageData
            The image/volume to be projected.

        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return
        
        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed
        '''

        return self.operator.direct(IM, out=out)

    def adjoint(self, DATA, out=None):
        '''Applies the adjoint of the operator, i.e. the backward projection.

        Parameters
        ----------
        DATA : AcquisitionData
            The projections/sinograms to be projected.

        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return
        
        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed
        '''
        return self.operator.adjoint(DATA, out=out)

    def calculate_norm(self):
        return self.operator.norm()

    def domain_geometry(self):
        return self.volume_geometry

    def range_geometry(self):
        return self.sinogram_geometry
