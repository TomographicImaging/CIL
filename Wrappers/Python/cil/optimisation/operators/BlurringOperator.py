#  Copyright 2020 United Kingdom Research and Innovation
#  Copyright 2020 The University of Manchester
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
from cil.optimisation.operators import LinearOperator
import cil

from scipy.ndimage import convolve, correlate

class BlurringOperator(LinearOperator):

    r'''BlurringOperator:  D: X -> X,  takes in a numpy array PSF representing
    a point spread function for blurring the image. The implementation is
    generic and naive simply using convolution.

        :param PSF: numpy array with point spread function of blur.
        :param geometry: ImageGeometry of ImageData to work on.

     '''

    def __init__(self, PSF, geometry):
        super(BlurringOperator, self).__init__(domain_geometry=geometry,
                                           range_geometry=geometry)
        if isinstance(PSF,np.ndarray):
            self.PSF = PSF
        else:
            raise TypeError('PSF must be a number array with same number of dimensions as geometry.')

        if not (isinstance(geometry,cil.framework.framework.ImageGeometry) or \
                isinstance(geometry,cil.framework.framework.AcquisitionGeometry)):
            raise TypeError('geometry must be an ImageGeometry or AcquisitionGeometry.')


    def direct(self,x,out=None):

        '''Returns D(x). The forward mapping consists of convolution of the
        image with the specified PSF. Here reflective boundary conditions
        are selected.'''

        if out is None:
            result = self.range_geometry().allocate()
            result.fill(convolve(x.as_array(),self.PSF, mode='reflect'))
            return result
        else:
            outarr = out.as_array()
            convolve(x.as_array(),self.PSF, output=outarr, mode='reflect')
            out.fill(outarr)
            return out
    
    def adjoint(self,x, out=None):

        '''Returns D^{*}(y). The adjoint of convolution is convolution with
        the PSF rotated by 180 degrees, or equivalently correlation by the PSF
        itself.'''

        if out is None:
            result = self.domain_geometry().allocate()
            result.fill(correlate(x.as_array(),self.PSF, mode='reflect'))
            return result
        else:
            outarr = out.as_array()
            correlate(x.as_array(),self.PSF, output=outarr, mode='reflect')
            out.fill(outarr)
            return out
