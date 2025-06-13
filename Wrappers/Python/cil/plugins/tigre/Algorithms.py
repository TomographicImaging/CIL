#  Copyright 2025 United Kingdom Research and Innovation
#  Copyright 2025 The University of Manchester
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

from cil.plugins.tigre import CIL2TIGREGeometry
import tigre.algorithms as algs
from cil.framework import ImageData
import logging
import numpy as np

from cil.framework.labels import AcquisitionDimension

log = logging.getLogger(__name__)


class tigre_algo_wrapper():
    
    def __init__(self, name=None,  initial=None, image_geometry=None,  data=None, niter=0,  **kwargs):
        """
        A wrapper for TIGRE algorithms, allowing the use of CIL geometries and data.
        
        Parameters
        ----------
        name : str
            Name of the TIGRE algorithm to use (e.g., 'ART', 'SART', 'SIRT', 'OSSART').
        initial : ImageData, optional
            Initial guess for the reconstruction. If None, a zero-initialized image is used.
        image_geometry : ImageGeometry
            The geometry of the image to be reconstructed.
        data : AcquisitionData
            The measured projection data.
        niter : int, default=0
            Number of iterations for the reconstruction algorithm.
        **kwargs : dict
            Additional keyword arguments passed to the TIGRE reconstruction algorithm.
            
        Returns
        -------
        ImageData
            The reconstructed image.
        quality : float
            Quality measures computed by the algorithm, if applicable. See the tigre algorithm documentation for details.
        Raises
        ------
        ValueError
            If `image_geometry` or `data` is None.
        Notes
        -----
        This class is designed to facilitate the use of TIGRE algorithms within the CIL framework,
        allowing for the use of CIL's `ImageGeometry` and `AcquisitionData` classes. It handles the conversion
        of CIL geometries to TIGRE geometries and prepares the data for the specified algorithm.
        The `name` parameter should match one of the available TIGRE algorithms. 
        Example
        -------
        >>> from cil.plugins.tigre import tigre_algo_wrapping 
        >>> algo = tigre_algo_wrapper(name='SART', initial=initial_image, image_geometry=image_geom, data=acquisition_data, niter=10)
        >>> reconstructed_image, quality = algo.process()
        
        """
        
        missing = []
        if image_geometry is None:
            missing.append("`image_geometry`")
        if data is None:
            missing.append("`data`")

        if missing:
            raise ValueError(f"You must pass {', '.join(missing)}")
        
        log.info("%s setting up tigre geometry", self.__class__.__name__)

        
        if initial is None:
            initial = image_geometry.allocate(0)

        self.tigre_initial = initial.copy().as_array()
        self.ig = image_geometry
        self.ag = data.geometry
        self.tigre_geom, self.tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            self.ig, self.ag)
        self.tigre_projections = data.as_array()
        
        if data.dimension_labels[0] != AcquisitionDimension.ANGLE: #TODO: Not sure when this is used/ if it is needed 
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=0)
            
        if self.tigre_geom.is2D:
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=1)
            self.tigre_initial = np.expand_dims(self.tigre_initial, axis=0)
            
        self.tigre_algo =  getattr(algs, name)
        self.niter = niter
        self.kwargs = kwargs

        log.info("%s configured", self.__class__.__name__)
        
    def process(self):
        """
        Run the specified TIGRE algorithm with the provided parameters.
        """
        
        log.info("%s passing to the tigre algorithm", self.__class__.__name__)
        
        result = self.tigre_algo(
            self.tigre_projections,
            self.tigre_geom,
            self.tigre_angles,
            init=self.tigre_initial,
            niter=self.niter,
            **self.kwargs
        )
        
        img = result[0] if isinstance(result, tuple) else result
        
        quality = result[1] if len(result) > 1 else None
        
        return ImageData(img, geometry=self.ig), quality