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

import numpy as np
from cil.framework import ImageData
from cil.optimisation.operators import LinearOperator

from cil.framework import ImageGeometry, AcquisitionGeometry, BlockGeometry

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
        :param dimension: 'prepend' (default) or 'append' channel dimension onto existing dimensions
                       
     '''
    
    def __init__(self, op, channels, dimension='prepend'):
        
        dom_op = op.domain_geometry()
        ran_op = op.range_geometry()
        
        geom_mc = []
        
        # Create multi-channel domain and range geometries: Clones of the
        # input single-channel geometries but with the specified number of
        # channels and additional dimension_label 'channel'.
        for geom in [dom_op,ran_op]:
            if dimension == 'prepend':
                new_dimension_labels = ['channel']+list(geom.dimension_labels)
            elif dimension == 'append':
                new_dimension_labels = list(geom.dimension_labels)+['channel']
            else:
                raise Exception("dimension must be either 'prepend' or 'append'")
            if isinstance(geom, ImageGeometry):
                
                geom_channels = geom.copy()
                geom_channels.channels = channels
                geom_channels.dimension_labels = new_dimension_labels

                geom_mc.append(geom_channels)
            elif isinstance(geom, AcquisitionGeometry):
                geom_channels = geom.copy()
                geom_channels.config.channels.num_channels = channels
                geom_channels.dimension_labels = new_dimension_labels

                geom_mc.append(geom_channels)
                    
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
                self.op.direct(x.get_slice(channel=k),cury)
                output.fill(cury.as_array(),channel=k)
            return output
        else:
            cury = self.op.range_geometry().allocate()
            for k in range(self.channels):
                self.op.direct(x.get_slice(channel=k),cury)
                out.fill(cury.as_array(),channel=k)
    
    def adjoint(self,x, out=None):
        
        '''Returns D^{*}(y)'''        
        
        # Loop over channels, extract single-channel data, apply single-channel
        # operator's adjoint method and fill into multi-channel output data set.
        if out is None:
            output = self.domain_geometry().allocate()
            cury = self.op.domain_geometry().allocate()
            for k in range(self.channels):
                self.op.adjoint(x.get_slice(channel=k),cury)
                output.fill(cury.as_array(),channel=k)
            return output
        else:
            cury = self.op.domain_geometry().allocate()
            for k in range(self.channels):
                self.op.adjoint(x.get_slice(channel=k),cury)
                out.fill(cury.as_array(),channel=k)
        
    def calculate_norm(self, **kwargs):
        
        '''Evaluates operator norm of DiagonalOperator'''
        
        return self.op.norm()

