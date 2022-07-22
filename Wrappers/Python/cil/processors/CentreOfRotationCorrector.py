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

from cil.processors.CofR_xcorrelation import CofR_xcorrelation
from cil.processors.CofR_image_sharpness import CofR_image_sharpness


class CentreOfRotationCorrector(object):
    """
    This class contains factory methods to create a CentreOfRotationCorrector object using the desired algorithm.
    """
       
    @staticmethod
    def xcorrelation(slice_index='centre', projection_index=0, ang_tol=0.1):
        r'''This creates a CentreOfRotationCorrector processor using the cross-correlation algorithm.

        For use on parallel-beam geometry it requires two projections 180 degree apart.

        Parameters
        ----------

        slice_index : int, str='centre', default='centre'
            An integer defining the vertical slice to run the algorithm on.

        projection_index : int
            An integer defining the first projection the algorithm will use. The second projection at 180 degrees will be located automatically.

        ang_tol : float, default=0.1
            The angular tolerance in degrees between the two input projections 180 degree gap

        Example
        -------
        >>> from cil.processors import CentreOfRotationCorrector
        >>> processor = CentreOfRotationCorrector.xcorrelation('centre')
        >>> processor.set_input(data)
        >>> data_centred = processor.get_ouput()

        Example
        -------
        >>> from cil.processors import CentreOfRotationCorrector
        >>> processor = CentreOfRotationCorrector.xcorrelation(slice_index=120)
        >>> processor.set_input(data)
        >>> processor.get_ouput(out=data)


        Example
        -------
        >>> from cil.processors import CentreOfRotationCorrector
        >>> import logging
        >>> logging.basicConfig(level=logging.WARNING)
        >>> cil_log_level = logging.getLogger('cil.processors')
        >>> cil_log_level.setLevel(logging.DEBUG)

        >>> processor = CentreOfRotationCorrector.xcorrelation(slice_index=120)
        >>> processor.set_input(data)
        >>> data_centred = processor.get_ouput()


        Note
        ----
        setting logging to 'debug' will give you more information about the algorithm progress



        '''

        processor = CofR_xcorrelation(slice_index, projection_index, ang_tol)
        return processor

    @staticmethod
    def image_sharpness(slice_index='centre', backend='tigre', tolerance=0.005, search_range=None, initial_binning=None, **kwargs):
        """
        This creates a CentreOfRotationCorrector processor which will find the centre by maximising the sharpness of a reconstructed slice.

        Can be used on single slice parallel-beam, and centre slice cone beam geometry. For use only with datasets that can be reconstructed with FBP/FDK.

        Parameters
        ----------

        slice_index : int, str='centre', default='centre'
            An integer defining the vertical slice to run the algorithm on.

        backend : str, default='tigre'
            The backend to use for the reconstruction, 'astra' or 'tigre'

        tolerance : float, default=1./200.
            The tolerance of the fit in pixels, the default is 1/200 of a pixel. This is a stopping criteria, not a statement of accuracy of the algorithm.

        search_range : int, default 0.25*pixels_num_h
            The range in pixels to search either side of the panel centre. If `None` a quarter of the width of the panel is used.  

        initial_binning : int, default=None
            The size of the bins for the initial grid. If `None` will bin the image to a step corresponding to <128 pixels. Note the fine search will be on unbinned data.


        **kwargs:
        FBP : Class
            Deprecated parameter: the FBP class imported from cil.plugins.[backend].FBP Please use 'backend' instead


        Example
        -------
        from cil.processors import CentreOfRotationCorrector

        processor = CentreOfRotationCorrector.image_sharpness('centre', 'tigre')
        processor.set_input(data)
        data_centred = processor.get_output()


        Example
        -------
        from cil.processors import CentreOfRotationCorrector

        processor = CentreOfRotationCorrector.image_sharpness(slice_index=120, 'astra')
        processor.set_input(data)
        processor.get_output(out=data)


        Note
        ----
        For best results data should be 360deg which leads to blurring with incorrect geometry.
        This method is unreliable on half-scan data with 'tuning-fork' style artifacts.

        """
        processor = CofR_image_sharpness(slice_index=slice_index, backend=backend, tolerance=tolerance, search_range=search_range, initial_binning=initial_binning, **kwargs)
        return processor
