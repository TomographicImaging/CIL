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
import glob
import importlib.metadata
import json
import logging
import numpy as np
import os
import sys
import warnings
from cil.framework.labels import AcquisitionDimension

log = logging.getLogger(__name__)

# TIGRE algorithms that force blocksize=1 and produce a bug with 3.1.3
SINGLE_ANGLE_ALGORITHMS = ('sart', 'sart_tv', 'asd_pocs', 'awasd_pocs', 'pcsd', 'aw_pcsd')

# TIGRE algorithms with a bug under NumPy >= 2 and 3.1.3.
NUMPY2_PROMOTION_ALGORITHMS = ('fista',)

# TIGRE algorithms that call `im3ddenoise` with potential multigpu single slice bug 
TV_DENOISE_ALGORITHMS = ('ista', 'fista', 'sart_tv', 'ossart_tv')

# Next release of tigre with potential fixes for single-angle projection and FISTA float64 bugs. The single-angle bug is fixed in master, but not yet released.
NEXT_TIGRE_RELEASE = (3, 1, 4)

TIGRE_SINGLE_ANGLE_FIX_RELEASE = NEXT_TIGRE_RELEASE
TIGRE_FISTA_FLOAT64_FIX_RELEASE = NEXT_TIGRE_RELEASE

# Names TIGRE has been packaged under.
TIGRE_DISTRIBUTIONS = ('tigre', 'pytigre')

_tigre_version_number = None


def _tigre_older_than(release):
    """
    Return True if the installed TIGRE predates `release`, given as a tuple of ints.

    Returns False when TIGRE is not installed, so that the bug checks below stay quiet rather than
    warning about a TIGRE that is not there.

    Note that TIGRE only bumps its version at release time, so a build from master reports the
    version of the last release, currently 3.1.3, even though master already contains every fix
    below. Anyone building TIGRE from source will therefore see warnings for bugs their TIGRE does
    not have.
    """
    global _tigre_version_number

    if algs is None:
        return False

    if _tigre_version_number is None:
        version = ''

        for name in TIGRE_DISTRIBUTIONS:
            try:
                version = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                # conda writes `conda-meta/<name>-<version>-<build>.json` for what it installed
                for path in glob.glob(os.path.join(sys.prefix, 'conda-meta', name + '-*.json')):
                    try:
                        with open(path) as record_file:
                            record = json.load(record_file)
                    except (OSError, ValueError):
                        continue

                    # the glob also catches packages whose name merely starts with `name`
                    if record.get('name') == name:
                        version = record.get('version', '')
                        break

            if version:
                break

        # keep the leading numeric components, so '3.1.3.dev0+g1234abc' gives (3, 1, 3) and an
        # unparseable version gives ()
        parts = []
        for part in version.split('.'):
            if not part.isdigit():
                break
            parts.append(int(part))

        _tigre_version_number = tuple(parts)

    return _tigre_version_number < release


def _has_tigre_single_angle_bug():
    """
    Detect the TIGRE bug that breaks any projection of a single angle.

    Reported upstream as CERN/TIGRE#744 and fixed by CERN/TIGRE#749 (commits 5f9a52691f and
    26d8e2e8ff, 2026-06-23), which is currently not in a release.

    Note this can be deleted when tigre is released with the fix, and CIL is updated to use that
    release.
    """
    return _tigre_older_than(TIGRE_SINGLE_ANGLE_FIX_RELEASE)


def _has_tigre_fista_float64_bug():
    """
    Detect the TIGRE bug that breaks FISTA under NumPy >= 2.

    Fixed upstream by CERN/TIGRE commit 1c8f53b ("Numpy 2.4 fixes in FISTA", 2026-08-14),
    which casts the square root to float32. The fix is not currently in a release.

    NumPy 1 promotes the expression the other way, so an old NumPy hides the bug whatever TIGRE is
    installed.

    Note this can be deleted when tigre is released with the fix, and CIL is updated to use that
    release. The bug is only triggered by FISTA, so we don't need to check other algorithms.
    """
    return (np.lib.NumpyVersion(np.__version__) >= '2.0.0'
            and _tigre_older_than(TIGRE_FISTA_FLOAT64_FIX_RELEASE))


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

        There are currently three known bugs in TIGRE that can affect the use of this wrapper. Each one is
        detected at initialisation and warned about, so you should not need to check for them yourself:
        1. Single-angle projection bug: Some algorithms (e.g., 'sart', 'sart_tv', 'asd_pocs', 'awasd_pocs', 'pcsd', 'aw_pcsd') project one angle at a time and cannot run with certain versions of TIGRE. If you encounter an error related to single-angle projection, please check the TIGRE version and consider using an ordered-subsets variant of the algorithm with blocksize > 1.
        2. FISTA float64 bug: The 'fista' algorithm upcasts its working volume to float64 under NumPy >= 2, which can lead to errors. If you encounter an error related to float64, please check the TIGRE version and consider using 'ista', which is unaffected.
        3. 2D TV denoising bug: The TV-based algorithms ('ista', 'fista', 'sart_tv', 'ossart_tv') use im3ddenoise, which can lead to issues when denoising a single-slice volume across multiple GPUs. If you encounter NaNs or unexpected results, please ensure that you are using a single GPU for 2D data. See
         https://github.com/CERN/TIGRE/issues/681. 


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

        self.tigre_algo = getattr(algs, algorithm_name)
        self.number_iterations = number_iterations
        self.kwargs = kwargs
        self.gpuids = self.kwargs.pop('gpuids', None)
        if self.gpuids is None:
            self.gpuids = GpuIds()
        log.info("Using GPU ids: %s", self.gpuids)

        self._warn_if_single_angle_bug(algorithm_name)
        self._warn_if_fista_float64_bug(algorithm_name)
        self._warn_if_2d_denoise_bug(algorithm_name)

        super(tigre_algo_wrapper, self).__init__(data, image_geometry=ig, backend='tigre')

        log.info("%s configured", self.__class__.__name__)

    def _warn_if_single_angle_bug(self, algorithm_name):
        """
        Warn if this configuration will hit the TIGRE single-angle bug, see
        `_has_tigre_single_angle_bug`. `run` fails deep inside TIGRE, so the warning gives the
        user something to search for before that happens.
        """
        if not _has_tigre_single_angle_bug():
            return

        if algorithm_name in SINGLE_ANGLE_ALGORITHMS:
            reason = "'{}' projects one angle at a time".format(algorithm_name)
        elif self.kwargs.get('blocksize') == 1:
            reason = "blocksize=1 projects one angle at a time"
        elif np.shape(self.tigre_angles)[0] == 1:
            reason = "the dataset holds a single angle"
        else:
            return

        warnings.warn(
            "This version of TIGRE cannot project a single angle, and {}, so `run` is expected "
            "to fail with 'only 0-dimensional arrays can be converted to Python scalars'. This "
            "is CERN/TIGRE#744, fixed upstream by CERN/TIGRE#749 but not in any TIGRE release "
            "up to v3.1.3. Update TIGRE, or use an ordered-subsets variant of the algorithm "
            "with blocksize > 1.".format(reason), UserWarning, stacklevel=3)

    def _warn_if_fista_float64_bug(self, algorithm_name):
        """
        Warn if this configuration will hit the TIGRE FISTA float64 bug, see
        `_has_tigre_fista_float64_bug`. The volume is only upcast at the end of an iteration, so
        a single iteration still runs and is not worth warning about.
        """
        if (algorithm_name not in NUMPY2_PROMOTION_ALGORITHMS or self.number_iterations < 2
                or not _has_tigre_fista_float64_bug()):
            return

        warnings.warn(
            "TIGRE's '{}' upcasts its working volume to float64 under NumPy {}, so `run` with "
            "number_iterations > 1 is expected to fail with 'Input data should be float32, not "
            "float64'. This is fixed upstream by CERN/TIGRE commit 1c8f53b but is not in any "
            "TIGRE release up to v3.1.3. Update TIGRE, or use 'ista', which is "
            "unaffected.".format(algorithm_name, np.__version__), UserWarning, stacklevel=3)

    def _warn_if_2d_denoise_bug(self, algorithm_name):
        """
        Warn if this configuration TV-denoises a single-slice volume across more than one GPU.

        The TV algorithms always call `im3ddenoise`, so with 2D data they hand `tv_proximal` a
        volume one slice deep, which TIGRE then splits over every GPU it was given. This is a
        known issue - CERN/TIGRE#681.

        """
        if algorithm_name not in TV_DENOISE_ALGORITHMS or not self.tigre_geom.is2D:
            return

        devices = getattr(self.gpuids, 'devices', None) or []
        if len(devices) < 2:
            return

        warnings.warn(
            "Potential issue with im3ddenoise for a 2D data on multiple GPUs: CERN/TIGRE#681 "
            "For safety,  pass a `gpuids` holding a single  device", UserWarning, stacklevel=3)

    def set_input(self, input):
        """
        Set the input data to run the reconstructor on. The geometry of the dataset must be compatible
        with the reconstructor.

        Called by the parent class during initialisation, and may be called afterwards to reuse the
        configured algorithm on a different dataset.

        Parameters
        ----------
        input : AcquisitionData
            A dataset with a compatible geometry
        """
        super().set_input(input)
        self.tigre_projections = self._prepare_projections(input)

    def _prepare_projections(self, data):
        """
        Return the data reshaped to the layout TIGRE expects.

        `as_array` returns a pointer and `expand_dims` returns a view, so this holds the user's
        buffer rather than a copy.
        """
        projections = data.as_array()

        if data.dimension_labels[0] != AcquisitionDimension.ANGLE:
            projections = np.expand_dims(projections, axis=0)

        if self.tigre_geom.is2D:
            projections = np.expand_dims(projections, axis=1)

        return projections

    def _get_tigre_initial(self):
        """
        Build a fresh, TIGRE-shaped float32 initial volume.

        This protects the user's `initial` and keeps repeated `run` calls independent of each other.

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
