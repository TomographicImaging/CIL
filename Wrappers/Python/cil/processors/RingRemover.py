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

class RingRemover(Processor):

    '''
        RingRemover Processor: Removes vertical stripes from a DataContainer(ImageData/AcquisitionData)
        using the algorithm in https://doi.org/10.1364/OE.17.008567

        Parameters
        ----------
        decNum : int
            Number of wavelet decompositions - increasing the number of decompositions, increases the strength of the ring removal
            but can alter the profile of the data

        wname : str
            Name of wavelet filter from pywt e.g. 'db1' -- 'db35', 'haar' - increasing the wavelet filter increases the strength of
            the ring removal, but also increases the computational effort

        sigma : float
            Damping parameter in Fourier space - increasing sigma, increases the size of artefacts which can be removed

        info : boolean
            Flag to enable print of ring remover end message

        Returns
        -------
        DataContainer
            Corrected ImageData/AcquisitionData 2D, 3D, multi-spectral 2D, multi-spectral 3D
    '''

    def __init__(self, decNum=4, wname='db10', sigma=1.5, info = True):

        kwargs = {'decNum': decNum,
                  'wname': wname,
                  'sigma': sigma,
                  'info': info}

        super(RingRemover, self).__init__(**kwargs)


    def check_input(self, dataset):
        if not ((isinstance(dataset, ImageData)) or
                (isinstance(dataset, AcquisitionData))):
            raise Exception('Processor supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')
        elif (dataset.geometry == None):
            raise Exception('Geometry is not defined.')
        else:
            return True

    def process(self, out = None):

        data = self.get_input()
        decNum = self.decNum
        wname = self.wname
        sigma = self.sigma
        info = self.info

        # acquisition geometry from sinogram
        geom = data.geometry

        # get channels, vertical info
        channels =  geom.channels
        vertical = geom.pixel_num_v

        # allocate datacontainer space
        out = 0.*data

        # for non multichannel data
        if 'channel' not in geom.dimension_labels:

            # for 3D data
            if 'vertical' in geom.dimension_labels:

                for i in range(vertical):
                    tmp_corrected = self._xRemoveStripesVertical(data.get_slice(vertical=i, force=True).as_array(), decNum, wname, sigma)
                    out.fill(tmp_corrected, vertical = i)

            # for 2D data
            else:
                tmp_corrected = self._xRemoveStripesVertical(data.as_array(), decNum, wname, sigma)
                out.fill(tmp_corrected)

        # for multichannel data
        else:

           # for 3D data
            if 'vertical' in geom.dimension_labels:

                for i in range(channels):

                    out_ch_i = out.get_slice(channel=i)
                    data_ch_i = data.get_slice(channel=i)

                    for j in range(vertical):
                        tmp_corrected = self._xRemoveStripesVertical(data_ch_i.get_slice(vertical=j, force=True).as_array(), decNum, wname, sigma)
                        out_ch_i.fill(tmp_corrected, vertical = j)

                    out.fill(out_ch_i.as_array(), channel=i)

                    if info:
                        print("Finish channel {}".format(i))

            # for 2D data
            else:
                for i in range(channels):
                        tmp_corrected = self._xRemoveStripesVertical(data.get_slice(channel=i).as_array(), decNum, wname, sigma)
                        out.fill(tmp_corrected, channel = i)
                        if info:
                            print("Finish channel {}".format(i))
        if info:
            print("Finish Ring Remover")

        return out


    def _xRemoveStripesVertical(self, ima, decNum, wname, sigma):

        ''' Ring removal algorithm via combined wavelet and fourier filtering
            code from https://doi.org/10.1364/OE.17.008567
            translated in Python
        Parameters
        ----------
        ima : ndarray
            2D image data

        decNum : int
            Number of wavelet decompositions - increasing the number of decompositions, increases the strength of the ring removal
            but can alter the profile of the data

        wname : str
            Name of wavelet filter from pywt e.g. 'db1' -- 'db35', 'haar' - increasing the wavelet filter increases the strength of
            the ring removal, but also increases the computational effort

        sigma : float
            Damping parameter in Fourier space - increasing sigma, increase the size of artefacts which can be removed

        Returns
        -------
        Corrected 2D sinogram data (Numpy Array)

        '''

        original_extent = [slice(None, ima.shape[0], None), slice(None, ima.shape[1], None)]

        # allocate cH, cV, cD
        Ch = [None]*decNum
        Cv = [None]*decNum
        Cd = [None]*decNum

        # wavelets decomposition
        for i in range(decNum):
            ima, (Ch[i], Cv[i], Cd[i]) = pywt.dwt2(ima,wname)

        # FFT transform of horizontal frequency bands
        for i in range(decNum):

            # use to axis=0, which correspond to the angles direction
            fCv = fftshift(fft(Cv[i], axis=0))
            my, mx = fCv.shape

            # damping of vertical stripe information
            damp = 1 - np.exp(-np.array([range(-int(np.floor(my/2)),-int(np.floor(my/2))+my)])**2/(2*sigma**2))
            fCv *= damp.T

            # inverse FFT
            Cv[i] = np.real(ifft(ifftshift(fCv), axis=0))

        # wavelet reconstruction
        nima = ima
        for i in range(decNum-1,-1,-1):
            nima = nima[0:Ch[i].shape[0],0:Ch[i].shape[1]]
            nima = pywt.idwt2((nima,(Ch[i],Cv[i],Cd[i])),wname)

        # if the original input is odd, the signal reconstructed with idwt2 will have one extra sample, which can be discarded
        nima = nima[original_extent[0], original_extent[1]]

        return nima
