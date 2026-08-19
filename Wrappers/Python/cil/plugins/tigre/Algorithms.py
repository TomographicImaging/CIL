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
    from tigre.utilities.gpu import GpuIds
except ModuleNotFoundError:
    algs = None
from cil.plugins.tigre import CIL2TIGREGeometry
from cil.framework import ImageData
import logging
import numpy as np
import warnings
from cil.framework.labels import AcquisitionDimension

log = logging.getLogger(__name__)

import weakref


class tigre_algo_wrapper(Reconstructor):

    def __init__(self, algorithm_name=None,  initial=None, image_geometry=None,  data=None, number_iterations=0,  **kwargs):
        """
        A wrapper for TIGRE algorithms, allowing the use of CIL geometries and data.

        Parameters
        ----------
        algorithm_name : str
            Name of the TIGRE algorithm to use (e.g., 'art', 'sart', 'sirt', 'ossart').
        initial : ImageData, optional
            Initial guess for the reconstruction. If None, a zero-initialized image is used.
        image_geometry : ImageGeometry, optional
            The geometry of the image to be reconstructed. If None, it is taken from `initial`,
            or otherwise from a default calculated from the geometry of `data`.
        data : AcquisitionData
            The measured projection data.
        number_iterations : int, default=0
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
        ModuleNotFoundError
            If TIGRE is not installed.
        ValueError
            If `data` is None.

        Notes
        -----
        This class is designed to facilitate the use of TIGRE algorithms within the CIL framework,
        allowing for the use of CIL's `ImageGeometry` and `AcquisitionData` classes. It handles the conversion
        of CIL geometries to TIGRE geometries and prepares the data for the specified algorithm.
        The `algorithm_name` parameter should match one of the available TIGRE algorithms for example: 'art', 'sirt', 'sart', 'ossart', 'cgls', 'lsmr', 'hybrid_lsqr', 'ista', 'fista', 'sart_tv', 'ossart_tv'.

        We are aware that running the TIGRE algorithms: ISTA, FISTA, SART_TV, OSSART_TV using 2D data can lead to
        incorrect results in the TV denoising step, particularly when using more than one GPU. See
        https://github.com/CERN/TIGRE/issues/681
        You can change the GPUs used by passing the `gpuids` keyword argument, for example:

        .. code-block:: python

            from tigre.utilities.gpu import GpuIds
            gpuids = GpuIds()
            gpuids.devices = [0]  # Specify the GPU device IDs you want to use
            algo = tigre_algo_wrapper(algorithm_name='fista', initial=initial_image, image_geometry=image_geom, data=acquisition_data, number_iterations=10, gpuids=gpuids)


        Example
        -------
        >>> from cil.plugins.tigre import tigre_algo_wrapper
        >>> algo = tigre_algo_wrapper(algorithm_name='sart', initial=initial_image, image_geometry=image_geom, data=acquisition_data, number_iterations=10)
        >>> reconstructed_image, quality = algo.run()

        """

        if algs is None:
            raise ModuleNotFoundError(
                "This plugin requires the additional package TIGRE\n"
                "Please install it via conda as tigre from the ccpi channel")

        if data is None:
            raise ValueError("`data` is required")
        if image_geometry is None and initial is None:
            image_geometry = data.geometry.get_ImageGeometry()
        elif image_geometry is None and initial is not None:
            image_geometry = initial.geometry

        log.info("%s setting up tigre geometry", self.__class__.__name__)

        self._initial = initial
        ig = image_geometry
        ag = data.geometry
        self.tigre_geom, self.tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            ig, ag)
        self.tigre_projections = data.as_array()

        # DEVELOPER NOTE: revisit this warning whenever the TIGRE version is updated.
        # It guards against CERN/TIGRE#681, an out-of-bounds read in the TV denoising step
        # for single-slice (2D) images, which corrupted the result when the image was split
        # across more than one GPU. A CUDA-side fix (`image_size[2] > 1` guards in
        # Common/CUDA/tv_proximal.cu) landed in CERN/TIGRE#699 and is present in TIGRE
        # v3.1.3, which is CIL's current minimum, so this warning is expected to be
        # obsolete. It is kept only because the fix has not yet been confirmed on
        # multi-GPU hardware. Once it has, remove this block, the matching paragraph in
        # the docstring above, and the `expect_warning` machinery in
        # test/test_PluginsTigre_Algorithms.py.
        if self.tigre_projections.ndim == 2:
            if any( a==algorithm_name for a in ['ista', 'fista', 'sart_tv', 'ossart_tv']):
                warnings.warn(
                    "We are aware that the TIGRE algorithms: ISTA, FISTA, SART_TV, OSSART_TV using 2D data can lead to incorrect results in the TV denoising step, particularly when using more than one GPU.", UserWarning, stacklevel=2)


        if data.dimension_labels[0] != AcquisitionDimension.ANGLE:
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=0)

        if self.tigre_geom.is2D:
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=1)

        self.tigre_algo = getattr(algs, algorithm_name)
        self.number_iterations = number_iterations
        self.kwargs = kwargs
        self.gpuids = self.kwargs.pop('gpuids', None)
        if self.gpuids is None:
            self.gpuids = GpuIds()
        log.info("Using GPU ids: %s", self.gpuids)

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

    def _get_tigre_initial(self):
        """
        Build a fresh, TIGRE-shaped float32 initial volume.

        TIGRE binds the array it is given (`self.res = init` in
        tigre/algorithms/iterative_recon_alg.py) and several algorithms update it in place,
        so this must return a private buffer each time: it protects the user's `initial` and
        keeps repeated `run` calls independent of each other.
        """
        if self._initial is None:
            return np.zeros(self.tigre_geom.nVoxel, dtype=np.float32)
        # astype always copies, so this is both the dtype guard and the defensive copy
        return self._initial.as_array().astype(np.float32).reshape(self.tigre_geom.nVoxel)

    def run(self, out=None):
        """
        Run the specified TIGRE algorithm with the provided parameters.

        Parameters
        ----------
        out : ImageData, optional
            Output image data to store the result. If None, a new ImageData object is created.
            Note that, unlike the other reconstructors in `cil.recon`, the result is returned
            whether or not `out` is passed, because `quality` is returned alongside it.

        Returns
        -------
        out : ImageData
            The reconstructed image data. This is `out` itself if `out` was passed.
        quality : float
            Quality measures computed by the tigre algorithm, if applicable.

        Notes
        -----
        `run` may be called repeatedly on the same object; each call restarts from `initial`.

        """

        log.info("%s passing to the tigre algorithm", self.__class__.__name__)
        tigre_initial = self._get_tigre_initial()
        result = self.tigre_algo(
            proj=self.tigre_projections,
            geo=self.tigre_geom,
            angles=self.tigre_angles,
            init=tigre_initial,
            niter=self.number_iterations,
            gpuids=self.gpuids,
            **self.kwargs
        )
        if isinstance(result, tuple):
            img = result[0]
            quality = result[1]
        else:
            img = result
            quality = None

        img = np.squeeze(img)

        if out is None:
            # TIGRE hands back a freshly allocated float32 array, so wrap it rather than
            # allocating a second volume and copying into it
            out = ImageData(img.astype(self._image_geometry.dtype, copy=False),
                            deep_copy=False, geometry=self._image_geometry)
        else:
            out.fill(img)

        log.info("%s completed", self.__class__.__name__)

        return out, quality
