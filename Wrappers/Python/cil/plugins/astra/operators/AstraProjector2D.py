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


from typing import Literal, Optional

from cil.framework.framework import (
    AcquisitionData,
    AcquisitionGeometry,
    DataContainer,
    ImageData,
    ImageGeometry,
)
from cil.optimisation.operators import LinearOperator
from cil.plugins.astra.processors import AstraBackProjector2D, AstraForwardProjector2D


class AstraProjector2D(LinearOperator):
    """
    AstraProjector2D configures and calls the ASTRA 2D Projectors for CPU and GPU.

    It is recommended to use this via the ProjectionOperator Class.

    Parameters
    ----------
    image_geometry
        A description of the area/volume to reconstruct

    acquisition_geometry
        A description of the acquisition data

    device
        The device to run on 'gpu' or 'cpu'

    Example
    -------
    >>> from cil.plugins.astra import AstraProjector2D
    >>> PO = AstraProjector2D(image.geometry, data.geometry)
    >>> forward_projection = PO.direct(image)
    >>> backward_projection = PO.adjoint(data)
    """

    def __init__(self, image_geometry:ImageGeometry, acquisition_geometry:AcquisitionGeometry, device:Literal["cpu", "gpu"]):

        super(AstraProjector2D, self).__init__(image_geometry, range_geometry=acquisition_geometry)

        self.fp = AstraForwardProjector2D(volume_geometry=image_geometry,
                                        sinogram_geometry=acquisition_geometry,
                                        proj_id = None,
                                        device=device)

        self.bp = AstraBackProjector2D(volume_geometry = image_geometry,
                                        sinogram_geometry = acquisition_geometry,
                                        proj_id = None,
                                        device = device)

    def direct(self, x:ImageData, out:Optional[ImageData]=None) -> Optional[ImageData]:
        """Apply the direct of the operator i.e. the forward projection.

        Parameters
        ----------
        x : ImageData
            The image/volume to be projected.

        out : ImageData, optional
           Fills the referenced ImageData with the processed data and suppresses the return

        Returns
        -------
        ImageData
            The processed data. Suppressed if `out` is passed
        """
        self.fp.set_input(x)
        return self.fp.get_output(out = out)

    def adjoint(self, x:AcquisitionData, out:Optional[DataContainer]=None) -> Optional[DataContainer]:
        """Apply the adjoint of the operator, i.e. the backward projection.

        Parameters
        ----------
        x : AcquisitionData
            The projections/sinograms to be projected.

        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return

        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed
        """
        self.bp.set_input(x)
        return self.bp.get_output(out = out)
