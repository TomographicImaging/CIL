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

from cil.processors.CofR_xcorr import CofR_xcorr
from cil.processors.CofR_sobel import CofR_sobel


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
        processor = CofR_xcorr(slice_index, projection_index, ang_tol)
        return processor

    @staticmethod
    def sobel(slice_index='centre', FBP=None, tolerance=0.005, search_range=None, initial_binning=None):
        r'''This creates a CentreOfRotationCorrector processor which will find the centre by maximising the sharpness of a reconstructed slice.

        Can be used on single slice parallel-beam, and centre slice cone beam geometry. For use only with datasets that can be reconstructed with FBP.

        :param slice_index: An integer defining the vertical slice to run the algorithm on.
        :type slice_index: int, str='centre', optional
        :param FBP: A CIL FBP class imported from cil.plugins.tigre or cil.plugins.astra  
        :type FBP: class
        :param tolerance: The tolerance of the fit in pixels, the default is 1/200 of a pixel. Note this is a stopping critera, not a statement of accuracy of the algorithm.
        :type tolerance: float, default = 0.001    
        :param search_range: The range in pixels to search either side of the panel centre. If `None` the width of the panel/4 is used. 
        :type search_range: int
        :param initial_binning: The size of the bins for the initial grid. If `None` will bin the image to a step corresponding to <128 pixels. Note the fine search will be on unbinned data.
        :type initial_binning: int
        :return: returns an AcquisitionData object with an updated AcquisitionGeometry
        :rtype: AcquisitionData
        '''
        processor = CofR_sobel(slice_index=slice_index, FBP=FBP, tolerance=tolerance, search_range=search_range, initial_binning=initial_binning)
        return processor
