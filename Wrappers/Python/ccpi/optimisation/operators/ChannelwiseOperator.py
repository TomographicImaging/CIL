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

from ccpi.framework.framework import ImageGeometry, AcquisitionGeometry

class ChannelwiseOperator(LinearOperator):
    
    r'''ChannelwiseOperator:  takes in a single-channel operator op and the 
    number of channels to be used, and creates a new multi-channel 
    ChannelwiseOperator, which will apply the operator op independently on 
    each channel for the number of channels specified.
                       
        :param op: Single-channel operator
        :param channels: Number of channels
                       
     '''
    
    def __init__(self, op, channels, channel_label='channel'):
        
        dom_op = op.domain_geometry()
        ran_op = op.range_geometry()
        
        if isinstance(dom_op, ImageGeometry):
            d = ImageGeometry(  
                            dom_op.voxel_num_x, 
                            dom_op.voxel_num_y, 
                            dom_op.voxel_num_z, 
                            dom_op.voxel_size_x, 
                            dom_op.voxel_size_y, 
                            dom_op.voxel_size_z, 
                            dom_op.center_x, 
                            dom_op.center_y, 
                            dom_op.center_z, 
                            channels,
                            dimension_labels=[channel_label] + dom_op.dimension_labels)
        elif isinstance(dom_op, AcquisitionGeometry):
            d = AcquisitionGeometry(
                                   dom_op.geom_type,
                                   dom_op.dimension, 
                                   dom_op.angles, 
                                   dom_op.pixel_num_h, 
                                   dom_op.pixel_size_h, 
                                   dom_op.pixel_num_v, 
                                   dom_op.pixel_size_v, 
                                   dom_op.dist_source_center, 
                                   dom_op.dist_center_detector, 
                                   channels,
                                   dimension_labels=[channel_label] + dom_op.dimension_labels)
        else:
            pass
        
        if isinstance(ran_op, ImageGeometry):
            r = ImageGeometry(  
                            ran_op.voxel_num_x, 
                            ran_op.voxel_num_y, 
                            ran_op.voxel_num_z, 
                            ran_op.voxel_size_x, 
                            ran_op.voxel_size_y, 
                            ran_op.voxel_size_z, 
                            ran_op.center_x, 
                            ran_op.center_y, 
                            ran_op.center_z, 
                            channels,
                            dimension_labels=[channel_label] + ran_op.dimension_labels)
        elif isinstance(ran_op, AcquisitionGeometry):
            r = AcquisitionGeometry(
                                   ran_op.geom_type,
                                   ran_op.dimension, 
                                   ran_op.angles, 
                                   ran_op.pixel_num_h, 
                                   ran_op.pixel_size_h, 
                                   ran_op.pixel_num_v, 
                                   ran_op.pixel_size_v, 
                                   ran_op.dist_source_center, 
                                   ran_op.dist_center_detector, 
                                   channels,
                                   dimension_labels=[channel_label] + ran_op.dimension_labels)
        else:
            pass
        
        super(ChannelwiseOperator, self).__init__(domain_geometry=d, 
                                           range_geometry=r)
        
        self.op = op
        self.channels = channels
        self.channel_label = channel_label

        
    def direct(self,x,out=None):
        
        '''Returns D(x)'''
        
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
        
        return self.op.calculate_norm()

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
    

    
    
    
    