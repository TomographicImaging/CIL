# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017-2020 UKRI-STFC
#   Copyright 2017-2020 University of Manchester

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

import numpy as np
from ccpi.optimisation.operators import LinearOperator

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
        self.PSF = PSF
        self.PSF_adjoint = np.rot90(PSF,2)

        
    def direct(self,x,out=None):
        
        '''Returns D(x). The forward mapping consists of convolution of the 
        image with the specified PSF. Here reflective boundary conditions 
        are selected.'''
        
        if out is None:
            result = self.range_geometry().allocate()
            result.fill(convolve(x.as_array(),self.PSF, mode='reflect'))
            return result
        else:
            convolve(x.as_array(),self.PSF, output=out.as_array(),mode='reflect')
    
    def adjoint(self,x, out=None):
        
        '''Returns D^{*}(y). The adjoint of convolution is convolution with 
        the PSF rotated by 180 degrees, or equivalently correlation by the PSF
        itself.'''
        
        if out is None:
            result = self.domain_geometry().allocate()
            result.fill(correlate(x.as_array(),self.PSF, mode='reflect'))
            return result
        else:
            correlate(x.as_array(),self.PSF, output=out.as_array(),mode='reflect')

if __name__ == '__main__':
    
    import matplotlib.pyplot as plt
    
    from ccpi.optimisation.algorithms import PDHG
    
    from ccpi.optimisation.operators import BlockOperator, Gradient
    from ccpi.optimisation.functions import ZeroFunction, MixedL21Norm, \
                                            BlockFunction, L2NormSquared
    from ccpi.framework import TestData
    import os
    import sys
    
    # Load in test image
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data_rgb = loader.load(TestData.PEPPERS)
    ig_rgb = data_rgb.geometry
    
    # Create gray version of image
    data_gray = 0.2989*data_rgb.subset(channel=0) + \
                0.5870*data_rgb.subset(channel=1) + \
                0.1140*data_rgb.subset(channel=2)
    ig_gray = data_gray.geometry
    
    # Display clean original image
    plt.figure(), plt.imshow(data_gray.as_array()), plt.gray(), plt.colorbar()
    
    # Parameters for point spread function PSF (size and std)
    ks          = 11; 
    ksigma      = 5.0;
    
    # Create 1D PSF and 2D as outer product, then normalise.
    w           = np.exp(-np.arange(-(ks-1)/2,(ks-1)/2+1)**2/(2*ksigma**2))
    w.shape     = (ks,1)
    PSF         = w*np.transpose(w)
    PSF         = PSF/(PSF**2).sum()
    PSF         = PSF/PSF.sum()
    
    # Display PSF as image
    plt.figure(), plt.imshow(PSF), plt.gray(), plt.colorbar()
    
    # Create blurring operator and apply to clean image to produce blurred,
    # and display.
    BOP = BlurringOperator(PSF,ig_gray)
    blurredimage = BOP.direct(data_gray)
    plt.figure(), plt.imshow(blurredimage.as_array()), plt.gray(), plt.colorbar()
    
    # Further apply adjoint for illustration and display.    
    adjointimage = BOP.adjoint(blurredimage)
    plt.figure(), plt.imshow(adjointimage.as_array()), plt.gray(), plt.colorbar()
    
    # Run dot test to check validity of adjoint.
    print(BOP.dot_test(BOP))
    
    # Specify total variation regularised least squares
    
    # Create operators
    op1 = Gradient(ig_gray, correlation=Gradient.CORRELATION_SPACE)
    op2 = BOP
    
    # Set regularisation parameter.
    alpha = 0.02
    
    # Create functions to be blocked with operators
    f1 = alpha * MixedL21Norm()
    f2 = 0.5 * L2NormSquared(b=blurredimage)
    
    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 
    
    # Create functions      
    f = BlockFunction(f1, f2) 
    g = ZeroFunction()
            
    # Compute operator Norm
    normK = operator.norm()
    
    # Primal & dual stepsizes
    sigma = 1
    tau = 1/(sigma*normK**2)
    
    # Setup and run the PDHG algorithm
    pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
    pdhg.max_iteration = 10000
    pdhg.update_objective_interval = 1
    pdhg.run(200,very_verbose=True)
    
    # Show results
    plt.figure(figsize=(20,5))
    plt.subplot(1,3,1)
    plt.imshow(data_gray.as_array(),vmin=0.0,vmax=1.0)
    plt.title('Ground Truth')
    plt.gray()
    plt.colorbar()
    plt.subplot(1,3,2)
    plt.imshow(blurredimage.as_array(),vmin=0.0,vmax=1.0)
    plt.title('Noisy and Masked Data')
    plt.gray()
    plt.colorbar()
    plt.subplot(1,3,3)
    plt.imshow(pdhg.get_output().as_array(),vmin=0.0,vmax=1.0)
    plt.title('TV Reconstruction')
    plt.gray()
    plt.colorbar()
    plt.show()
