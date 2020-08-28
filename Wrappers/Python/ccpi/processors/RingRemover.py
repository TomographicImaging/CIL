# -*- coding: utf-8 -*-

from scipy.fftpack import fftshift, ifftshift, fft, ifft
import numpy as np
import pywt
from ccpi.framework import DataProcessor, ImageData, AcquisitionData

class RingRemover(DataProcessor):
    
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
        ImageData/AcquisitionData
            Corrected 2D, 3D, multi-spectral 2D, multi-spectral 3D    
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
        corrected_data = 0.*data 
        out = corrected_data
        
        # allocate numpy space
        tmp_data_array = corrected_data.as_array()
        
        # sinogram numpy array
        data_array = data.as_array()
        
        if channels == 1:        
            if vertical>0:
                for i in range(vertical):
                    J = self.xRemoveStripesVertical(data_array[i], decNum, wname, sigma)
                    tmp_data_array[i] = J                
            else:
                J = self.xRemoveStripesVertical(data_array, decNum, wname, sigma)
                out.fill(J)        
        else:
            if vertical>0:
                for i in range(channels):
                    for j in range(vertical):
                        J = self.xRemoveStripesVertical(data_array[i,j], decNum, wname, sigma)
                        out.fill(J, channel=i, vertical = j)
                    if info:
                        print("Finish channel {}".format(i))                    
            else:
                for i in range(channels):
                        J = self.xRemoveStripesVertical(data_array[i], decNum, wname, sigma)
                        out.fill(J, channel = i)
                        if info:
                            print("Finish channel {}".format(i))
        if info:
            print("Finish Ring Remover") 
                    
        return out
          
    def xRemoveStripesVertical(self,ima, decNum, wname, sigma):
        
        ''' Code from https://doi.org/10.1364/OE.17.008567 
            translated in Python
                
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
            my,mx = fCv.shape
            
            # damping of vertical stripe information
            damp = 1 - np.exp(-np.array([range(-int(np.floor(my/2)),-int(np.floor(my/2))+my)])**2/(2*sigma**2))
            fCv*=damp.T
             
            # inverse FFT          
            Cv[i] = np.real(ifft(ifftshift(fCv), axis=0))
                                                                    
        # wavelet reconstruction
        nima = ima
        for i in range(decNum-1,-1,-1):
            nima = nima[0:Ch[i].shape[0],0:Ch[i].shape[1]]
            nima = pywt.idwt2((nima,(Ch[i],Cv[i],Cd[i])),wname)
            
        return nima      

