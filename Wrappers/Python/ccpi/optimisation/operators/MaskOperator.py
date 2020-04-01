# -*- coding: utf-8 -*-
#  CCP in Tomographic Imaging (CCPi) Core Imaging Library (CIL).

#   Copyright 2017 UKRI-STFC
#   Copyright 2017 University of Manchester

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
from ccpi.framework import ImageData
from ccpi.optimisation.operators import DiagonalOperator

class MaskOperator(DiagonalOperator):
    
    r'''MaskOperator:  D: X -> X,  takes in a DataContainer or subclass 
    thereof, mask, of datatype Boolean where a True represents a value to be
    kept and False a value to be lost/set to zero. Maps an element of 
    :math:`x\in X` onto the element :math:`y \in X,  y = mask*x`, 
    where * denotes elementwise multiplication.
                       
        :param mask: DataContainer with True/False elements
                       
     '''
    
    def __init__(self, mask):
        super(MaskOperator, self).__init__(mask)

        

if __name__ == '__main__':
    
    from ccpi.framework import ImageGeometry

    M = 3
    ig = ImageGeometry(M, M)
    x = ig.allocate('random_int')
    diag = ig.allocate('random_int')
    
    # Print what each ImageData is
    print(x.as_array())
    print(diag.as_array())
    
    
    
    # Set up example DiagonalOperator
    D = DiagonalOperator(diag)
    
    # Apply direct and check whether result equals diag*x as expected.
    z = D.direct(x)
    print(z.as_array())
    print((diag*x).as_array())
    
    # Apply adjoint and check whether results equals diag*(diag*x) as expected.
    y = D.adjoint(z)
    print(y.as_array())
    print((diag*(diag*x)).as_array())
    
    import numpy as np 
    import numpy                          
    import matplotlib.pyplot as plt
    
    from ccpi.optimisation.algorithms import PDHG
    
    from ccpi.optimisation.operators import BlockOperator, Identity, Gradient
    from ccpi.optimisation.functions import ZeroFunction, L1Norm, \
                          MixedL21Norm, BlockFunction, L2NormSquared,\
                              KullbackLeibler
    from ccpi.framework import TestData
    import os
    import sys
    
    # user supplied input
    if len(sys.argv) > 1:
        which_noise = int(sys.argv[1])
    else:
        which_noise = 2
    print ("Applying {} noise")
    
    if len(sys.argv) > 2:
        method = sys.argv[2]
    else:
        method = '1'
        
    method = '0'
    print ("method ", method)
    
    
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data = loader.load(TestData.SHAPES)
    ig = data.geometry
    ag = ig
    
    mask = ig.allocate()
    mask = mask + 1
    amask = mask.as_array()
    amask[140:160,10:90] = 0.0
    amask[70:130,140:160] = 0.0
    amask[15:50,180:240] = 0.0
    amask[95:105,180:295] = 0.0
    
    MO = MaskOperator(mask)
    
    # Create noisy data. 
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
    
    # Show Ground Truth and Noisy Data
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.show()
    
    
    # Regularisation Parameter depending on the noise distribution
    if noise == 's&p':
        alpha = 0.8
    elif noise == 'poisson':
        alpha = 1
    elif noise == 'gaussian':
        alpha = .3
    
    # fidelity
    if noise == 's&p':
        f2 = L1Norm(b=noisy_data)
    elif noise == 'poisson':
        f2 = KullbackLeibler(noisy_data)
    elif noise == 'gaussian':
        f2 = 0.5 * L2NormSquared(b=noisy_data)
    
    if method == '0':
    
        # Create operators
        op1 = Gradient(ig, correlation=Gradient.CORRELATION_SPACE)
        op2 = MO #Identity(ig, ag)
    
        # Create BlockOperator
        operator = BlockOperator(op1, op2, shape=(2,1) ) 
    
        # Create functions      
        f = BlockFunction(alpha * MixedL21Norm(), f2) 
        g = ZeroFunction()
        
    else:
        
        operator = Gradient(ig)
        f =  alpha * MixedL21Norm()
        g = f2
            
        
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
    plt.imshow(data.subset(channel=0).as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.imshow(noisy_data.subset(channel=0).as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(pdhg.get_output().subset(channel=0).as_array())
    plt.title('TV Reconstruction')
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), data.as_array()[int(ig.shape[0]/2),:], label = 'GTruth')
    plt.plot(np.linspace(0,ig.shape[1],ig.shape[1]), pdhg.get_output().as_array()[int(ig.shape[0]/2),:], label = 'TV reconstruction')
    plt.legend()
    plt.title('Middle Line Profiles')
    plt.show()
    
        
        