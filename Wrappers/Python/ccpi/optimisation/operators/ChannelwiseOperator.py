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
from ccpi.framework import ImageData
from ccpi.optimisation.operators import LinearOperator

from ccpi.framework import ImageGeometry, AcquisitionGeometry, BlockGeometry

class ChannelwiseOperator(LinearOperator):
    
    r'''ChannelwiseOperator:  takes in a single-channel operator op and the 
    number of channels to be used, and creates a new multi-channel 
    ChannelwiseOperator, which will apply the operator op independently on 
    each channel for the number of channels specified.
    
    ChannelwiseOperator supports simple operators as input but not 
    BlockOperators. Typically if such behaviour is desired, it can be achieved  
    by creating instead a BlockOperator of ChannelwiseOperators.
                       
        :param op: Single-channel operator
        :param channels: Number of channels
                       
     '''
    
    def __init__(self, op, channels):
        
        dom_op = op.domain_geometry()
        ran_op = op.range_geometry()
        
        geom_mc = []
        
        # Create multi-channel domain and range geometries: Clones of the
        # input single-channel geometries but with the specified number of
        # channels and additional dimension_label 'channel'.
        for geom in [dom_op,ran_op]:
            if isinstance(geom, ImageGeometry):
                geom_mc.append(
                    ImageGeometry(  
                    geom.voxel_num_x, 
                    geom.voxel_num_y, 
                    geom.voxel_num_z, 
                    geom.voxel_size_x, 
                    geom.voxel_size_y, 
                    geom.voxel_size_z, 
                    geom.center_x, 
                    geom.center_y, 
                    geom.center_z, 
                    channels,
                    dimension_labels=['channel'] + dom_op.dimension_labels))
            elif isinstance(geom, AcquisitionGeometry):
                geom_mc.append(
                    AcquisitionGeometry(
                       geom.geom_type,
                       geom.dimension, 
                       geom.angles, 
                       geom.pixel_num_h, 
                       geom.pixel_size_h, 
                       geom.pixel_num_v, 
                       geom.pixel_size_v, 
                       geom.dist_source_center, 
                       geom.dist_center_detector, 
                       channels,
                       dimension_labels=['channel'] + dom_op.dimension_labels))
            elif isinstance(geom,BlockGeometry):
                raise Exception("ChannelwiseOperator does not support BlockOperator as input. Consider making a BlockOperator of ChannelwiseOperators instead.")
            else:
                pass
        
        super(ChannelwiseOperator, self).__init__(domain_geometry=geom_mc[0], 
                                           range_geometry=geom_mc[1])
        
        self.op = op
        self.channels = channels
        
    def direct(self,x,out=None):
        
        '''Returns D(x)'''
        
        # Loop over channels, extract single-channel data, apply single-channel
        # operator's direct method and fill into multi-channel output data set.
        if out is None:
            output = self.range_geometry().allocate()
            cury = self.op.range_geometry().allocate()
            for k in range(self.channels):
                self.op.direct(x.subset(channel=k),cury)
                output.fill(cury.as_array(),channel=k)
            return output
        else:
            cury = self.op.range_geometry().allocate()
            for k in range(self.channels):
                self.op.direct(x.subset(channel=k),cury)
                out.fill(cury.as_array(),channel=k)
    
    def adjoint(self,x, out=None):
        
        '''Returns D^{*}(y)'''        
        
        # Loop over channels, extract single-channel data, apply single-channel
        # operator's adjoint method and fill into multi-channel output data set.
        if out is None:
            output = self.domain_geometry().allocate()
            cury = self.op.domain_geometry().allocate()
            for k in range(self.channels):
                self.op.adjoint(x.subset(channel=k),cury)
                output.fill(cury.as_array(),channel=k)
            return output
        else:
            cury = self.op.domain_geometry().allocate()
            for k in range(self.channels):
                self.op.adjoint(x.subset(channel=k),cury)
                out.fill(cury.as_array(),channel=k)
        
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of DiagonalOperator'''
        
        return self.op.norm()

if __name__ == '__main__':
    
    from ccpi.optimisation.operators import DiagonalOperator

    M = 3
    channels = 4
    ig = ImageGeometry(M, M, channels=channels)
    igs = ImageGeometry(M, M)
    x = ig.allocate('random',seed=100)
    diag = igs.allocate('random',seed=101)
    
    D = DiagonalOperator(diag)
    C = ChannelwiseOperator(D,channels)
    
    y = C.direct(x)
    
    y2 = ig.allocate()
    C.direct(x,y2)
    
    print(y.subset(channel=2).as_array())
    print(y2.subset(channel=2).as_array())
    print((diag*x.subset(channel=2)).as_array())
    
    z = C.adjoint(y)
    
    z2 = ig.allocate()
    C.adjoint(y,z2)
    
    print(z.subset(channel=2).as_array())
    print(z2.subset(channel=2).as_array())
    print((diag*(diag*x.subset(channel=2))).as_array())
    