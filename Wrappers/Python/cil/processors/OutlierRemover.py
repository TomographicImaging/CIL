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

    def check_input(self, dataset):
        if not (isinstance(dataset, AcquisitionData)):
            raise Exception('Processor supports only following data types:\n' +
                            '- Acquisition Data\n')
        elif (dataset.geometry == None):
            raise Exception('Geometry is not defined.')
        else:
            return True

    
    def process(self, out=None):
        data = self.get_input()
        diff = self.diff
        radius = self.radius
        mode = self.mode

        if not diff or not diff > 0:
            raise ValueError(f'diff parameter must be greater than 0. Value provided was {diff}')

        if not radius or not radius > 0:
            raise ValueError(f'radius parameter must be greater than 0. Value provided was {radius}')
        
        if mode not in ['bright', 'dark']:
            raise ValueError("Mode must be either 'bright' or 'dark'")
        
        data = self.get_input()
        if out is None:
            out = data.copy()
        elif id(out) != id(data):
            np.copyto(out.array, data.array)
        
        data_array = out.as_array()

        for i in range(data_array.shape[0]):
            median = median_filter(data_array[i], size=radius)
            if mode == 'bright':
                data_array[i] = np.where((data_array[i] - median) > diff, median, data_array[i])
            else:  # mode == 'dark'
                data_array[i] = np.where(median - data_array[i] > diff, median, data_array[i])
        
        return out
