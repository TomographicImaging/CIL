#  Copyright 2024 United Kingdom Research and Innovation
#  Copyright 2024 The University of Manchester
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

from cil.framework import Processor, DataContainer, AcquisitionData,\
 AcquisitionGeometry, ImageGeometry, ImageData
from cil.utilities.display import show2D
import numpy
import logging

log = logging.getLogger(__name__)

class FluxNormaliser(Processor):
    '''Flux normalisation based on float or region of interest

    This processor reads in a AcquisitionData and normalises it based on
    a float or array of float values, or a region of interest.

    Input: AcquisitionData
    Parameter: 2D projection with flat field (or stack)
               2D projection with dark field (or stack)
    Output: AcquisitionDataSet
    '''

    def __init__(self, flux=None, roi=None, tolerance=1e-5):
            kwargs = {
                    'flux'  : flux,
                    'roi' : roi,
                    'roi_slice' : None,
                    # very small number. Used when there is a division by zero
                    'tolerance'   : tolerance
                    }
            super(FluxNormaliser, self).__init__(**kwargs)
            
    def check_input(self, dataset):
        
        if not issubclass(type(dataset), DataContainer):
            raise TypeError("Expected DataContainer or subclass, found {}"
                            .format(type(dataset)))
        if self.roi is not None:
            if self.flux is not None:
                raise ValueError("Please specify either flux or roi, not both")
            else:
                if isinstance(self.roi,dict):
                    slc = [slice(None)]*len(data.shape)
                    axes=[]
                    for r in self.roi:
                        if (r == 'horizontal' or r == 'vertical') and (r in data.dimension_labels):
                            if not all(isinstance(i, int) for i in self.roi[r]):
                                raise ValueError("roi values must be int, found {}"
                                .format(str(type(self.roi[r]))))
                            else:
                                ax = data.get_dimension_axis(r)
                                slc[ax] = slice(self.roi[r][0],self.roi[r][1])
                                axes.append(ax)
                        else:
                            raise ValueError("roi must be 'horizontal' or 'vertical', and in data.dimension_labels, found {}"
                            .format(str(r)))

                    self.flux = np.mean(data.array[tuple(slc)],axis=tuple(axes))

                else:
                    raise TypeError("roi must be a dictionary, found {}"
                    .format(str(type(self.roi))))

        flux_size = (numpy.shape(self.flux))
        if len(flux_size) > 0:
            data_size = numpy.shape(dataset.geometry.angles)
            if data_size != flux_size:
                raise ValueError("Flux must be a scalar or array with length \
                                    \n = data.geometry.angles, found {} and {}"
                                    .format(flux_size, data_size))
            
        return True

    def show_roi(self, angle='range'):
        if angle=='range':
            plot_slice_roi_angle(angle=0, ax=231)
            plot_slice_roi_angle(angle=len(data.geometry.angles)//2, ax=232)
            plot_slice_roi_angle(angle=len(data.geometry.angles)-1, ax=233)

            plt.subplot(212)
            plt.plot(self.flux)
            plt.xlabel('Angle index')
            plt.ylabel('Mean intensity in roi')

        else:
            data = self.get_input()
            plt.imshow(data.array[1,:,:], cmap='gray')
            plt.plot([0,120],[0,120],'--k')

    def plot_slice_roi_angle(angle, ax):
        data_slice = data.get_slice(angle=angle)
        plt.subplot(ax)
        plt.imshow(data_slice.array, cmap='gray',aspect='equal', origin='lower')

        h = data_slice.dimension_labels[0]
        v = data_slice.dimension_labels[1]
        plt.plot([self.roi[h][0],self.roi[h][0]], [self.roi[v][0],self.roi[v][1]],'--r')
        plt.plot([self.roi[h][1],self.roi[h][1]], [self.roi[v][0],self.roi[v][1]],'--r')

        plt.plot([self.roi[h][0],self.roi[h][1]], [self.roi[v][0],self.roi[v][0]],'--r')
        plt.plot([self.roi[h][0],self.roi[h][1]], [self.roi[v][1],self.roi[v][1]],'--r')

        plt.xlabel(h)
        plt.ylabel(v)
        plt.title('Angle = ' + str(angle))

        


    def process(self, out=None):
        
        data = self.get_input()

        if out is None:
            out = data.copy()

        flux_size = (numpy.shape(self.flux))
        
        proj_axis = data.get_dimension_axis('angle')
        slice_proj = [slice(None)]*len(data.shape)
        slice_proj[proj_axis] = 0
        
        f = self.flux
        for i in range(numpy.shape(data)[proj_axis]):
            if len(flux_size) > 0:
                f = self.flux[i]

            slice_proj[proj_axis] = i
            with numpy.errstate(divide='ignore', invalid='ignore'):
                out.array[tuple(slice_proj)] = data.array[tuple(slice_proj)]/f
                        
        out.array[ ~ numpy.isfinite( out.array )] = self.tolerance

        return out
