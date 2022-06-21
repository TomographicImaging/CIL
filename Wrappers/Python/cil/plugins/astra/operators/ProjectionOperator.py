# -*- coding: utf-8 -*-
#  Copyright 2019 - 2022 United Kingdom Research and Innovation
#  Copyright 2019 - 2022 The University of Manchester
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


from cil.framework import DataOrder
from cil.optimisation.operators import LinearOperator, ChannelwiseOperator
from cil.plugins.astra.operators import AstraProjector3D
from cil.plugins.astra.operators import AstraProjector2D

class ProjectionOperator(LinearOperator):

    """
    ProjectionOperator configures and calls appropriate ASTRA Projectors for your dataset.


    Parameters
    ----------

    image_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    acquisition_geometry : AcquisitionGeometry
        A description of the acquisition data

    device : string, default='gpu'
        'gpu' will run on a compatible CUDA capable device using the ASTRA 3D CUDA Projectors, 'cpu' will run on CPU using the ASTRA 2D CPU Projectors


    Example
    -------

    >>> from cil.plugins.astra import ProjectionOperator
    >>> PO = ProjectionOperator(image.geometry, data.geometry)
    >>> foward_projection = PO.direct(image)
    >>> backward_projection = PO.adjoint(data)

    Notes
    -----
    For multichannel data the ProjectionOperator will broadcast across all channels.
    """

    def __init__(self, image_geometry, acquisition_geometry, device='gpu'):
        
        super(ProjectionOperator, self).__init__(domain_geometry=image_geometry, range_geometry=acquisition_geometry)

        DataOrder.check_order_for_engine('astra', image_geometry)
        DataOrder.check_order_for_engine('astra', acquisition_geometry) 

        self.volume_geometry = image_geometry
        self.sinogram_geometry = acquisition_geometry

        sinogram_geometry_sc = acquisition_geometry.subset(channel=0)
        volume_geometry_sc = image_geometry.subset(channel=0)

        if device == 'gpu':
            operator = AstraProjector3D(volume_geometry_sc, sinogram_geometry_sc)
        elif self.sinogram_geometry.dimension == '2D':
            operator = AstraProjector2D(volume_geometry_sc, sinogram_geometry_sc,  device=device)
        else:
            raise NotImplementedError("Cannot process 3D data without a GPU")

        if acquisition_geometry.channels > 1: 
            operator_full = ChannelwiseOperator(operator, self.sinogram_geometry.channels, dimension='prepend')
            self.operator = operator_full
        else:
            self.operator = operator

    def direct(self, IM, out=None):
        return self.operator.direct(IM, out=out)
    
    def adjoint(self, DATA, out=None):
        return self.operator.adjoint(DATA, out=out)
    
    def calculate_norm(self):
        return self.operator.norm()    

    def domain_geometry(self):
        return self.volume_geometry
    
    def range_geometry(self):
        return self.sinogram_geometry
