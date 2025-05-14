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

from cil.optimisation.algorithms import Algorithm
from cil.plugins.tigre import CIL2TIGREGeometry
import tigre.algorithms as algs
from cil.framework import ImageData
import logging

log = logging.getLogger(__name__)
import numpy as np


class ART(Algorithm):
    r"""  TODO:
    """

    def __init__(self, initial=None, operator=None, data=None, blocksize = None, nonneg = False, **kwargs):

        super(ART, self).__init__(**kwargs)

        self.set_up(initial=initial, operator=operator, data=data, blocksize = blocksize, nonneg = nonneg)


    def set_up(self, initial=None, operator=None, data=None, blocksize = None, nonneg=False):
        '''Set up the algorithm'''
        
        log.info("%s setting up", self.__class__.__name__)
        
        warning = 0
        if operator is None:
            warning += 1
            msg = "an `operator`"
        if data is None:
            warning += 10
            if warning > 10:
                msg += " and `data`"
            else:
                msg = "`data`"
        if blocksize is None:
            warning += 10
            if warning > 20:
                msg += " and `blocksize`"
            else:
                msg = "`blocksize`"
        if warning > 0:
            raise ValueError(f'You must pass {msg} to the ART algorithm' )
        
        if initial is None:
            initial = operator.domain_geometry().allocate(0)
            
        
        self.initial = initial.copy()
        self.operator = operator # TODO: is this needed or just an image and acquisition geometry 
        self.ig = operator.domain_geometry()
        self.ag = operator.range_geometry()
        self.data = data
        self.tigre_geom, self.tigre_angles = CIL2TIGREGeometry.getTIGREGeometry(
            self.ig, self.ag)
        self.tigre_projections = data.as_array()
        self.blocksize = blocksize

        self.tigre_alg = algs.iterative_recon_alg.IterativeReconAlg( self.tigre_projections, self.tigre_geom, self.tigre_angles, niter=0, blocksize= self.blocksize)
        
        self.configured = True
        log.info("%s configured", self.__class__.__name__)
    def update(self):
        self.tigre_alg.art_data_minimizing()

    def get_output(self):
        r""" Returns the current solution. 
        
        Returns
        -------
        DataContainer
            The current solution 
             
        """
        return ImageData(self.tigre_alg.getres(), geometry=self.ig) # TODO: Note that this only works for CT problems at the moment

    def update_objective(self):
        r""" Appends the current objective value to the list of previous objective values

        .. math:: \frac{1}{2}\|A x - b\|^{2}

        """
        self.loss.append(0.5*np.sum(((self.operator.direct(self.get_output()).as_array() - self.data.as_array())**2))) #TODO: this could possibly be done more efficiently - TIGRE might calculate it internally 



class OSSART(ART):

    def __init__(self, initial=None, operator=None, data=None, blocksize = None, nonneg = False, **kwargs):

        super(OSSART, self).__init__(initial=initial, operator=operator, data=data, blocksize = blocksize, nonneg = nonneg, **kwargs)
        
class SIRT(ART):
    
    def __init__(self, initial=None, operator=None, data=None, nonneg = False, **kwargs):

        blocksize=len(operator.range_geometry().angles)
        
        super(SIRT, self).__init__(initial=initial, operator=operator, data=data, blocksize = blocksize, nonneg = nonneg, **kwargs)
      
class SART(ART):
    
    def __init__(self, initial=None, operator=None, data=None, nonneg = False, **kwargs):

        blocksize=1
        
        super(SART, self).__init__(initial=initial, operator=operator, data=data, blocksize = blocksize, nonneg = nonneg, **kwargs)
    