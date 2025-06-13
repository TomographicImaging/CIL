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

import numpy as np
from cil.optimisation.algorithms import Algorithm
from cil.plugins.tigre import CIL2TIGREGeometry
import tigre.algorithms as algs
from cil.framework import ImageData
import logging
from tigre.utilities.Ax import Ax
from tigre.utilities.im3Dnorm import im3DNORM

from cil.framework.labels import AcquisitionDimension

log = logging.getLogger(__name__)


class ART(Algorithm):

    r"""
    Algebraic Reconstruction Technique (ART) implementation using the TIGRE backend.

    This class provides an interface to perform iterative image reconstruction using the ART algorithm.
    It leverages the TIGRE library for GPU-accelerated computations and supports configurable parameters
    such as block size and non-negativity constraints.

    Parameters
    ----------
    initial : ImageData, optional
            Initial guess for the reconstruction. If None, a zero-initialized image is used.
    image_geometry : ImageGeometry
            The geometry of the image to be reconstructed.
    data : AcquisitionData
            The measured projection data.
    blocksize : int
            Number of projections to use per iteration (subset size).
    noneg : bool, default=True
            If True, enforces non-negativity constraint on the reconstructed image.
    **kwargs : dict
            Additional keyword arguments passed to the TIGRE reconstruction algorithm.
    """

    def __init__(self, initial=None, image_geometry=None,  data=None, blocksize=None, noneg=True, **kwargs):

        update_objective_interval = kwargs.pop('update_objective_interval', 1)
        super(ART, self).__init__(
            update_objective_interval=update_objective_interval)

        self.set_up(initial=initial, image_geometry=image_geometry,
                    data=data, blocksize=blocksize, noneg=noneg, **kwargs)

    def set_up(self, initial=None, image_geometry=None,  data=None, blocksize=None, noneg=False, **kwargs):
        '''Set up the algorithm'''

        log.info("%s setting up", self.__class__.__name__)

        if initial is None:
            initial = image_geometry.allocate(0)

        tigre_initial = initial.copy().as_array()
        self.ig = image_geometry
        self.ag = data.geometry
        self.tigre_geom, self.tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            self.ig, self.ag)
        self.tigre_projections = data.as_array()
        
        if data.dimension_labels[0] != AcquisitionDimension.ANGLE: #TODO: Not sure when this is used/ if it is needed 
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=0)
            
        if self.tigre_geom.is2D:
            self.tigre_projections = np.expand_dims(self.tigre_projections, axis=1)
            tigre_initial = np.expand_dims(tigre_initial, axis=0)
            
        self.tigre_alg = algs.iterative_recon_alg.IterativeReconAlg(
            self.tigre_projections, self.tigre_geom, self.tigre_angles, init=tigre_initial, niter=0, blocksize=blocksize, noneg=noneg, **kwargs)

        self.configured = True
        log.info("%s configured", self.__class__.__name__)

    def update(self):
        """
        Performs one iteration of the ART algorithm using the TIGRE backend.
        """

        self.tigre_alg.art_data_minimizing()

    def get_output(self):
        r""" Returns the current solution. 

        Returns
        -------
        ImageData
            The current estimate of the reconstructed image.

        """
        return ImageData(self.tigre_alg.getres(), geometry=self.ig)

    def update_objective(self):
        r""" 
        Computes and stores the current value of the objective function.

        The objective function is defined as:

        .. math:: \frac{1}{2}\|A x - b\|^{2}


        where :math:`A` is the system matrix, :math:`x` is the current image estimate,
        and :math:`b` is the measured projection data


        """
        self.loss.append(im3DNORM(
            self.tigre_alg.proj - Ax(self.tigre_alg.res, self.tigre_geom,
                                     self.tigre_angles, "Siddon", gpuids=self.tigre_alg.gpuids), 2
        ))


class OSSART(ART): # TODO: This is just an alias of the ART parent algorithm - do we want them both? 

    r"""
    Ordered Subsets Simultaneous Algebraic Reconstruction Technique (OS-SART) from the Tigre library.

    This subclass of ART implements the OS-SART algorithm, which accelerates convergence
    by dividing the projection data into subsets (blocks) and updating the image using
    each subset sequentially within an iteration.

    Parameters
    ----------
    initial : ImageData, optional
        Initial guess for the reconstruction.
    image_geometry : ImageGeometry
        Geometry of the image to be reconstructed.
    data : AcquisitionData
        Measured projection data.
    blocksize : int
        Number of projections to use per subset.
    noneg : bool, default=True
        Enforce non-negativity constraint.
    **kwargs : dict
        Additional parameters for the TIGRE algorithm.
    """


    def __init__(self, initial=None, image_geometry=None, data=None, blocksize=None, noneg=True, **kwargs):
        
        # Collect missing required parameters
        missing = []
        if image_geometry is None:
            missing.append("`image_geometry`")
        if data is None:
            missing.append("`data`")
        if blocksize is None:
            missing.append("`blocksize`")

        if missing:
            raise ValueError(f"You must pass {', '.join(missing)} to the OSSART algorithm")

        super(OSSART, self).__init__(initial=initial, image_geometry=image_geometry,
                                    data=data, blocksize=blocksize, noneg=noneg, **kwargs)


class SIRT(ART):
    """
    Simultaneous Iterative Reconstruction Technique (SIRT) from the Tigre library. 

    This subclass of ART implements the SIRT algorithm, which uses all projections
    in each iteration (i.e., full blocksize). It is known for its stability and
    smooth convergence, especially in noisy data scenarios.

    Parameters
    ----------
    initial : ImageData, optional
            Initial guess for the reconstruction.
    image_geometry : ImageGeometry
            Geometry of the image to be reconstructed.
    data : AcquisitionData
            Measured projection data.
    noneg : bool, default=True
            Enforce non-negativity constraint.
    **kwargs : dict
            Additional parameters for the TIGRE algorithm.
    """

    def __init__(self, initial=None, image_geometry=None, data=None, noneg=True, **kwargs):
        
        # Collect missing required parameters
        missing = []
        if image_geometry is None:
            missing.append("`image_geometry`")
        if data is None:
            missing.append("`data`")

        if missing:
            raise ValueError(f"You must pass {', '.join(missing)} to the SIRT algorithm")

        blocksize = len(data.geometry.angles)

        super(SIRT, self).__init__(initial=initial, image_geometry=image_geometry,
                                   data=data, blocksize=blocksize, noneg=noneg, **kwargs)


class SART(ART):
    """ 
    Simultaneous Algebraic Reconstruction Technique (SART) from the Tigre library.

    This subclass of ART implements the SART algorithm, which updates the image
    using one projection at a time (i.e., blocksize = 1). 

    Parameters
    ----------
    initial : ImageData, optional
            Initial guess for the reconstruction.
    image_geometry : ImageGeometry
            Geometry of the image to be reconstructed.
    data : AcquisitionData
            Measured projection data.
    noneg : bool, default=True
            Enforce non-negativity constraint.
    **kwargs : dict
            Additional parameters for the TIGRE algorithm.
    """

    def __init__(self, initial=None, image_geometry=None, data=None, noneg=True, **kwargs):

        # Collect missing required parameters
        missing = []
        if image_geometry is None:
            missing.append("`image_geometry`")
        if data is None:
            missing.append("`data`")

        if missing:
            raise ValueError(f"You must pass {', '.join(missing)} to the SART algorithm")
        
        super(SART, self).__init__(initial=initial, image_geometry=image_geometry,
                                   data=data, blocksize=1, noneg=noneg, **kwargs)
