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

from cil.recon import Reconstructor
try:
    import tigre.algorithms as algs
    from cil.plugins.tigre import CIL2TIGREGeometry
except ImportError:
    raise ImportError("TIGRE is not installed. Please install it to use this module.")
from cil.framework import ImageData
import logging
import numpy as np
import warnings
from cil.framework.labels import AcquisitionDimension

log = logging.getLogger(__name__)

try:
    from tigre.utilities.gpu import GpuIds
    has_gpu_sel = True
except ModuleNotFoundError:
    has_gpu_sel = False
    
import weakref
    
class tigre_algo_wrapper(Reconstructor):

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
        The `name` parameter should match one of the available TIGRE algorithms for example: 'art', 'sirt', 'sart', 'ossart', 'cgls', 'lsmr', 'hybrid_lsqr', 'ista', 'fista', 'sart_tv', 'ossart_tv'.

        Note
        ----
        We are aware that running the TIGRE algorithms: ISTA, FISTA, SART_TV, OSSART_TV using 2D data can lead to incorrect restults in the TV denoising step, particularly when using more than one GPU. https://github.com/CERN/TIGRE/issues/681
        You can change the gpuids by passing the `gpuids` keyword argument, for example:
        ```python
        from tigre.utilities.gpu import GpuIds
        gpuids = GpuIds()
        gpuids.devices = [0]  # Specify the GPU device IDs you want to use
        algo = tigre_algo_wrapper(name='fista', initial=initial_image, image_geometry=image_geom, data=acquisition_data, niter=10, gpuids=gpuids)
        ```
        
        
        Example
        -------
        >>> from cil.plugins.tigre import tigre_algo_wrapping 
        >>> algo = tigre_algo_wrapper(name='SART', initial=initial_image, image_geometry=image_geom, data=acquisition_data, niter=10)
        >>> reconstructed_image, quality = algo.run()

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
        ig = image_geometry
        ag = data.geometry
        self.tigre_geom, self.tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            ig, ag)
        self.tigre_geom.check_geo(self.tigre_angles)
        self.tigre_projections = data.as_array()
        
        if self.tigre_projections.ndim == 2:
            if any( a==name for a in ['ista', 'fista', 'sart_tv', 'ossart_tv']):
                warnings.warn(
                    "We are aware that the TIGRE algorithms: ISTA, FISTA, SART_TV, OSSART_TV using 2D data can lead to incorrect results in the TV denoising step, particularly when using more than one GPU.", UserWarning, stacklevel=2)


        if data.dimension_labels[0] != AcquisitionDimension.ANGLE:
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=0)

        if self.tigre_geom.is2D:
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=1)
            self.tigre_initial = np.expand_dims(self.tigre_initial, axis=0)
            
        self.tigre_algo = getattr(algs, name)
        self.niter = niter
        self.kwargs = kwargs
        if has_gpu_sel:
            self.gpuids = self.kwargs.pop('gpuids', None)
            if self.gpuids is None: 
                self.gpuids =  GpuIds()
            log.info("Using GPU ids:", self.gpuids)
            
        self._input = None  

        super(tigre_algo_wrapper, self).__init__(data, image_geometry=ig, backend='tigre')

        log.info("%s configured", self.__class__.__name__)

    def set_input(self, input):
        """
        When called by the parent class during initialisation, sets the input data to run the reconstructor on. The geometry of the dataset must be compatible with the reconstructor.
        When called after initialisation, raises NotImplementedError as changing the input is not currently supported.
        Parameters
        ----------
        input : AcquisitionData
            A dataset with a compatible geometry
        """
        if self._input is None:
            if input.geometry != self.acquisition_geometry:
                raise ValueError ("Input not compatible with configured reconstructor. Initialise a new reconstructor with this geometry")
            else:
                self._input = weakref.ref(input)
                
        else:
            raise NotImplementedError("Setting the input after initialisation is not currently supported.")
    
    def run(self, out=None):
        """
        Run the specified TIGRE algorithm with the provided parameters.

        Parameters
        ----------
        out : ImageData, optional
            Output image data to store the result. If None, a new ImageData object is created.

        Returns
        -------
        out : ImageData
            The reconstructed image data.
        quality : float
            Quality measures computed by the tigre algorithm, if applicable.

        """

        log.info("%s passing to the tigre algorithm", self.__class__.__name__)
        if has_gpu_sel:
            result = self.tigre_algo(
                proj=self.tigre_projections,
                geo=self.tigre_geom,
                angles=self.tigre_angles,
                init=self.tigre_initial,
                niter=self.niter,
                gpuids=self.gpuids,
                **self.kwargs
            )
        else:
            result = self.tigre_algo(
                proj=self.tigre_projections,
                geo=self.tigre_geom,
                angles=self.tigre_angles,
                init=self.tigre_initial,
                niter=self.niter,
                **self.kwargs
            )

        img = result[0] if isinstance(result, tuple) else result

        quality = result[1] if len(result) > 1 else None

        if out is None:
            out = self._image_geometry.allocate(0)

        out.fill(img)

        log.info("%s completed", self.__class__.__name__)

        return out, quality
