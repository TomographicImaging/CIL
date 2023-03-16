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


from cil.optimisation.operators import LinearOperator
from cil.plugins.astra.processors import AstraForwardProjector3D, AstraBackProjector3D

class AstraProjector3D(LinearOperator):
    """
    AstraProjector3D configures and calls the ASTRA 3D Projectors for GPU. This works with 2D or 3D datasets.
    It is recommended to use this via the ProjectionOperator Class.

    Parameters
    ----------

    image_geometry : ImageGeometry
        A description of the area/volume to reconstruct

    acquisition_geometry : AcquisitionGeometry
        A description of the acquisition data

    Example
    -------
    >>> from cil.plugins.astra import AstraProjector3D
    >>> PO = AstraProjector3D(image.geometry, data.geometry)
    >>> forward_projection = PO.direct(image)
    >>> backward_projection = PO.adjoint(data)
    """

    def __init__(self, image_geometry, acquisition_geometry):
       
        super(AstraProjector3D, self).__init__(domain_geometry=image_geometry, range_geometry=acquisition_geometry)
                    
        self.sinogram_geometry = acquisition_geometry 
        self.volume_geometry = image_geometry         
        
        self.fp = AstraForwardProjector3D(volume_geometry=image_geometry, sinogram_geometry=acquisition_geometry)       
        self.bp = AstraBackProjector3D(volume_geometry=image_geometry, sinogram_geometry=acquisition_geometry)
                      
    def direct(self, x, out=None):
        '''Applies the direct of the operator i.e. the forward projection.
        
        Parameters
        ----------
        x : ImageData
            The image/volume to be projected.

        out : DataContainer, optional
           Fills the referenced DataContainer with the processed data and suppresses the return
        
        Returns
        -------
        DataContainer
            The processed data. Suppressed if `out` is passed
        '''

        self.fp.set_input(x)
        temp= self.fp.get_output(out = out)
        self.fp.__dict__['input']= None
        return temp
    
    def adjoint(self, x, out=None):
        '''Applies the adjoint of the operator, i.e. the backward projection.

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
        '''

        self.bp.set_input(x)  
        temp= self.bp.get_output(out = out)
        self.bp.__dict__['input']= None
        return temp
