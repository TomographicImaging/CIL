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
#from ccpi.optimisation.operators import DiagonalOperator

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
    
    from ccpi.framework import ImageGeometry, ImageData, \
                            AcquisitionGeometry, BlockDataContainer, AcquisitionData
    
    import numpy as np 
    import numpy                          
    import matplotlib.pyplot as plt
    
    from ccpi.optimisation.algorithms import CGLS
    from ccpi.optimisation.operators import BlockOperator, Gradient, Identity
    from ccpi.framework import TestData
           
    import os, sys
    
    loader = TestData(data_dir=os.path.join(sys.prefix, 'share','ccpi'))
    data = loader.load(TestData.SHAPES)
    ig = data.geometry
    ag = ig
    
    noisy_data = ImageData(TestData.random_noise(data.as_array(), mode = 'gaussian', seed = 1))
    #noisy_data = ImageData(data.as_array())
    
    mask = ig.allocate()
    mask = mask + 1
    amask = mask.as_array()
    amask[160:180,60:80] = 0.0
    
    MO = MaskOperator(mask)
    
    noisy_data = MO.direct(noisy_data)
    
    # Show Ground Truth and Noisy Data
    plt.figure(figsize=(10,10))
    plt.subplot(2,1,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.colorbar()
    plt.subplot(2,1,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy Data')
    plt.colorbar()
    plt.show()
    
    # Setup and run the regularised CGLS algorithm  (Tikhonov with Gradient)
    x_init = ig.allocate() 
    alpha = 2
    op = Gradient(ig)
    
    A = MO
    
    block_op = BlockOperator( A, alpha * op, shape=(2,1))
    block_data = BlockDataContainer(noisy_data, op.range_geometry().allocate())
       
    cgls = CGLS(x_init=x_init, operator = block_op, data = block_data)
    cgls.max_iteration = 200
    cgls.update_objective_interval = 5
    cgls.run(200, verbose = True)
    
    # Show results
    plt.figure(figsize=(20,10))
    plt.subplot(3,1,1)
    plt.imshow(data.as_array())
    plt.title('Ground Truth')
    plt.subplot(3,1,2)
    plt.imshow(noisy_data.as_array())
    plt.title('Noisy')
    plt.subplot(3,1,3)
    plt.imshow(cgls.get_output().as_array())
    plt.title('Regularised GGLS with Gradient')
    plt.show()
    
        
        