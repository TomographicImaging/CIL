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

from cil.optimisation.operators import DiagonalOperator

class MaskOperator(DiagonalOperator):
    
    r'''MaskOperator:  D: X -> X,  takes in a DataContainer or subclass 
    thereof, mask, with True or 1.0 representing a value to be
    kept and False or 0.0 a value to be lost/set to zero. Maps an element of 
    :math:`x\in X` onto the element :math:`y \in X,  y = mask*x`, 
    where * denotes elementwise multiplication.
                       
        :param mask: DataContainer of datatype bool or with 1/0 elements
                       
     '''
    
    def __init__(self, mask):
        # Special case of DiagonalOperator which is the superclass of
        # MaskOperator, so simply instanciate a DiagonalOperator with mask.
        super(MaskOperator, self).__init__(mask)
        self.mask = self.diagonal
        
        

if __name__ == '__main__':
                      
    import matplotlib.pyplot as plt
    
    from cil.optimisation.algorithms import PDHG
    
    from cil.optimisation.operators import BlockOperator, Gradient
    from cil.optimisation.functions import ZeroFunction, L1Norm, \
                          MixedL21Norm, BlockFunction, L2NormSquared,\
                              KullbackLeibler
    from cil.framework import TestData
    import os
    import sys
    
    # Specify which which type of noise to use.    
    which_noise = 0
    print ("which_noise ", which_noise)
    
    # Load in test image
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','cil'))
    data = loader.load(TestData.SHAPES)
    ig = data.geometry
    ag = ig
    
    # Create mask with four rectangles to be masked, set up MaskOperator    
    mask = ig.allocate(True,dtype=np.bool)
    amask = mask.as_array()
    amask[140:160,10:90] = False
    amask[70:130,140:160] = False
    amask[15:50,180:240] = False
    amask[95:105,180:295] = False
    
    MO = MaskOperator(mask)
    
    # Create noisy and masked data: First add noise, then mask the image with 
    # MaskOperator.
    noises = ['gaussian', 'poisson', 's&p']
    noise = noises[which_noise]
    if noise == 's&p':
        n1 = TestData.random_noise(data.as_array(), mode = noise, salt_vs_pepper = 0.9, amount=0.2)
    elif noise == 'poisson':
        scale = 5
        n1 = TestData.random_noise( data.as_array()/scale, mode = noise, seed = 10)*scale
    elif noise == 'gaussian':
        n1 = TestData.random_noise(data.as_array(), mode = noise, seed = 10)
    else:
        raise ValueError('Unsupported Noise ', noise)
    noisy_data = ig.allocate()
    noisy_data.fill(n1)
    
    noisy_data = MO.direct(noisy_data)    
    
    # Regularisation Parameter depending on the noise distribution
    if noise == 's&p':
        alpha = 0.8
    elif noise == 'poisson':
        alpha = 1.0
    elif noise == 'gaussian':
        alpha = .3
    
    # Choose data fidelity dependent on noise type.
    if noise == 's&p':
        f2 = L1Norm(b=noisy_data)
    elif noise == 'poisson':
        f2 = KullbackLeibler(noisy_data)
    elif noise == 'gaussian':
        f2 = 0.5 * L2NormSquared(b=noisy_data)
    
    # Create operators
    op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
    op2 = MO

    # Create BlockOperator
    operator = BlockOperator(op1, op2, shape=(2,1) ) 

    # Create functions      
    f = BlockFunction(alpha * MixedL21Norm(), f2) 
    g = ZeroFunction()
            
    # Compute operator Norm
    normK = operator.norm()
    
    # Primal & dual stepsizes
    sigma = 1
    tau = 1/(sigma*normK**2)
    
    # Setup and run the PDHG algorithm
    pdhg = PDHG(f=f,g=g,operator=operator, tau=tau, sigma=sigma)
    pdhg.max_iteration = 2000
    pdhg.update_objective_interval = 100
    pdhg.run(2000)
    
    # Show results
    plt.figure(figsize=(20,5))
    plt.subplot(1,4,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(pdhg.get_output().as_array())
    plt.title('TV Reconstruction')
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), noisy_data.as_array()[int(ig.shape[0]/2),:], label = 'Noisy and masked')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
