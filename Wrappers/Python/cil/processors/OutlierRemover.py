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

from scipy.fftpack import fftshift, ifftshift, fft, ifft
import numpy as np
import pywt
from cil.framework import Processor, ImageData, AcquisitionData
from scipy.ndimage import median_filter


class OutlierRemover(Processor):

    '''
        OutlierRemover Processor: Removes pixel values that are found to be outliers from a DataContainer(ImageData/AcquisitionData)

        Parameters
        ----------
        data : ImageData or AcquisitionData
            The input data to be processed

        diff: float
            Pixel value difference threshold. A pixel is considered an outlier if its
            value differs from the local median by more than this amount.

        radius : int
            Size of the neighbourhood used to compute the local median.

        mode: string
            Type of outlier to remove. Options are 'bright' to remove bright outliers, 'dark' to remove dark outliers,

        Returns
        -------
        DataContainer
            Corrected ImageData/AcquisitionData 3D numpy.ndarray
    '''

    def __init__(self, diff=50.0, radius=1, mode='bright'):

        kwargs = {'diff': diff,
                  'radius': radius,
                  'mode': mode}

        super(OutlierRemover, self).__init__(**kwargs)

    def check_input(self, dataset, diff, radius, mode):
        if not diff or not diff > 0:
            raise ValueError(f'diff parameter must be greater than 0. Value provided was {diff}')

        if not radius or not radius > 0:
            raise ValueError(f'radius parameter must be greater than 0. Value provided was {radius}')

        params = {'diff': diff, 'radius': radius, 'mode': mode}
        #ps.run_compute_func(OutliersFilter.compute_function, dataset.shape[0], dataset.shared_array, params)

        return dataset
    
    def process(self, out=None):
        data = self.get_input()
        diff = self.diff
        radius = self.radius
        mode = self.mode

        median = median_filter(data.as_array(), size=2*radius+1)