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

from cil.framework import ImageGeometry, AcquisitionGeometry, BlockGeometry
from cil.optimisation.operators import LinearOperator


class ChannelwiseOperator(LinearOperator):

    r'''The ChannelwiseOperator  takes in an operator  and the number of channels to be used, and creates a new multi-channel
    `ChannelwiseOperator`, which will apply the operator  independently on each channel for the number of channels specified.

    ChannelwiseOperator does not currently support BlockOperators. Typically, if such behaviour is desired, it can be achieved
    by creating instead a BlockOperator of ChannelwiseOperators.

    Parameters
    ----------
    op : Operator
        Single-channel operator
    channels : int
        Number of channels
    dimension : str, optional
        'prepend' (default) or 'append' channel dimension onto existing dimensions
        
    Example
    --------
    In this example, we create a ChannelwiseOperator that applies the same diagonal operator to each channel of a 2 channel image data.
    
    >>> M = 3
    >>> channels = 2
    >>> ig = ImageGeometry(M, M, channels=channels)
    >>> single_ig = ImageGeometry(M, M)
    >>> x = ImageData( np.stack( [np.ones((M,M)),  2*np.ones((M,M))] , axis=0), geometry=ig )
    >>> diag = ImageData(np.array(range(M*M), dtype=np.float64).reshape((M,M)), geometry=single_ig)
    >>> D = DiagonalOperator(diag)
    >>> C = ChannelwiseOperator(D,channels)
    >>> y = C.direct(x)
    >>> print('The original image data is:')
    >>> print(x.as_array())
    The original image data is:
    [[[1. 1. 1.]
    [1. 1. 1.]
    [1. 1. 1.]]
    [[2. 2. 2.]
    [2. 2. 2.]
    [2. 2. 2.]]]
    >>> print('The channel wise operator multiplies each channel element wise by:')
    >>> print(diag.as_array())
    The channel wise operator multiplies each channel element wise by:
    [[0. 1. 2.]
    [3. 4. 5.]
    [6. 7. 8.]]
    >>> print('The result of applying the channel wise operator is:')
    >>> print(y.as_array())
    The result of applying the channel wise operator is:
    [[[ 0.  1.  2.]
    [ 3.  4.  5.]
    [ 6.  7.  8.]]
    [[ 0.  2.  4.]
    [ 6.  8. 10.]
    [12. 14. 16.]]]
    
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
        '''
        Returns :math:`D(x)` where :math:`D` is the ChannelwiseOperator and :math:`x` is the input data.
        
        Parameters
        ----------
        x : DataContainer
            Input data
        out : DataContainer, optional
            Output data, if not provided a new DataContainer will be created
            
        Returns
        -------
        out : DataContainer
            Output data
        '''
        # Loop over channels, extract single-channel data, apply single-channel
        # operator's direct method and fill into multi-channel output data set.
        if out is None:
            out = self.range_geometry().allocate()
        cury = self.op.range_geometry().allocate()
        for k in range(self.channels):
            self.op.direct(x.get_slice(channel=k),cury)
            out.fill(cury.as_array(),channel=k)
        return out

    def adjoint(self, x, out=None):
        '''Returns :math:`D^{*}(x)` where :math:`D` is the ChannelwiseOperator and :math:`x` is the input data.
        
        Parameters
        ----------
        x : DataContainer
            Input data
        out : DataContainer, optional
            Output data, if not provided a new DataContainer will be created
    
        Returns
        -------
        out : DataContainer
            Output data
        
        '''
        # Loop over channels, extract single-channel data, apply single-channel
        # operator's adjoint method and fill into multi-channel output data set.
        if out is None:
            out = self.domain_geometry().allocate()
        cury = self.op.domain_geometry().allocate()
        for k in range(self.channels):
            self.op.adjoint(x.get_slice(channel=k),cury)
            out.fill(cury.as_array(),channel=k)
        return out

    def calculate_norm(self, **kwargs):
        '''Evaluates operator norm of ChannelWiseOperator'''
        return self.op.norm()
