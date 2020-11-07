# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi 
#   (Collaborative Computational Project in Tomographic Imaging), with 
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
 
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ccpi.processors.CofR_xcorr import CofR_xcorr

class CentreOfRotationCorrector(object):
    """
    This class contains factory methods to create a CentreOfRotationCorrector object using the desired algorithm.
    """

    @staticmethod
    def xcorr(slice_index='centre', projection_index=0, ang_tol=0.1):
        r'''This creates a CentreOfRotationCorrector processor using the cross-correlation algorithm.

        For use on parallel-beam geometry it requires two projections 180 degree apart.

        :param slice_index: An integer defining the vertical slice to run the algorithm on.
        :type slice_index: int, str='centre', optional
        :param projection_index: An integer defining the first projection the algorithm will use. The second projection at 180 degrees will be located automatically.
        :type projection_index: int, optional
        :param ang_tol: The angular tolerance in degrees between the two input projections 180degree gap
        :type ang_tol: float, optional
        :return: returns an AcquisitionData object with an updated AcquisitionGeometry
        :rtype: AcquisitionData
        '''
        proccessor = CofR_xcorr(slice_index, projection_index, ang_tol)
        return proccessor
