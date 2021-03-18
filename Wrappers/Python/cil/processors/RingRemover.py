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

from scipy.fftpack import fftshift, ifftshift, fft, ifft
import numpy as np
import pywt
from cil.framework import Processor, ImageData, AcquisitionData

class RingRemover(Processor):
    
    '''
        RingRemover Processor: Removes vertical stripes from a DataContainer(ImageData/AcquisitionData) 
        the algorithm in https://doi.org/10.1364/OE.17.008567

    '''
        
    def __init__(self, decNum, wname, sigma, info = True):
        
        '''
    
        Parameters
        ----------
        decNum : number of wavelet decompositions
        
        wname : (str) name of wavelet filter from pywt 
            Example: 'db1' -- 'db35', 'haar'
    
        sigma : Damping parameter in Fourier space.
        
        info : Prints ring remover end message 
        
        Returns
        -------
        Corrected ImageData/AcquisitionData 2D, 3D,
                multi-spectral 2D, multi-spectral 3D    
        '''             
        
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
                    tmp_corrected = self.xRemoveStripesVertical(data.subset(vertical=i, force=True).as_array(), decNum, wname, sigma) 
                    out.fill(tmp_corrected, vertical = i)  
            
            # for 2D data
            else:
                tmp_corrected = self.xRemoveStripesVertical(data.as_array(), decNum, wname, sigma)
                out.fill(tmp_corrected)        
        
        # for multichannel data        
        else:
            
           # for 3D data
            if 'vertical' in geom.dimension_labels:
                
                for i in range(channels):
                    
                    out_ch_i = out.subset(channel=i)
                    data_ch_i = data.subset(channel=i)
                    
                    for j in range(vertical):
                        tmp_corrected = self.xRemoveStripesVertical(data_ch_i.subset(vertical=j, force=True).as_array(), decNum, wname, sigma)
                        out_ch_i.fill(tmp_corrected, vertical = j)
                        
                    out.fill(out_ch_i.as_array(), channel=i) 
                    
                    if info:
                        print("Finish channel {}".format(i))                    
                                       
            # for 2D data                        
            else:
                for i in range(channels):
                        tmp_corrected = self.xRemoveStripesVertical(data.subset(channel=i).as_array(), decNum, wname, sigma)
                        out.fill(tmp_corrected, channel = i)
                        if info:
                            print("Finish channel {}".format(i))
        if info:
            print("Finish Ring Remover") 
                    
        return out

          
    def xRemoveStripesVertical(self,ima, decNum, wname, sigma):
        
        ''' Code from https://doi.org/10.1364/OE.17.008567 
            translated in Python
                            
        Returns
        -------
        Corrected 2D sinogram data (Numpy Array)
        
        '''              
                            
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
            
        return nima      