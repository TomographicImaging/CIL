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
import matplotlib.pyplot as plt
from scipy import stats

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
                    if not all (r in dataset.dimension_labels for r in self.roi):
                        raise ValueError("roi must be 'horizontal' or 'vertical', found '{}'"
                                        .format(str(self.roi)))

                    slc = [slice(None)]*len(dataset.shape)
                    axes=[]
                    roi_list = list(dataset.dimension_labels)
                    
                    # loop through all dimensions in the dataset apart from angle
                    roi_list.remove('angle')
                    for r in roi_list:
                        # check the dimension is in the user specified roi
                        if r in self.roi:
                            # only allow horizontal and vertical roi
                            if (r == 'horizontal' or r == 'vertical'):
                                # check indices are ints
                                if not all(isinstance(i, int) for i in self.roi[r]):
                                    raise TypeError("roi values must be int, found {} and {}"
                                    .format(str(type(self.roi[r][0])), str(type(self.roi[r][1]))))
                                # check indices are in range
                                elif (self.roi[r][0] >= self.roi[r][1]) or (self.roi[r][0] < 0) or self.roi[r][1] > dataset.get_dimension_size(r):
                                    raise ValueError("roi values must be start > stop and between 0 and {}, found start={} and stop={} for direction '{}'"
                                    .format(str(dataset.get_dimension_size(r)), str(self.roi[r][0]), str(self.roi[r][1]), r ))
                                # create slice
                                else:
                                    ax = dataset.get_dimension_axis(r)
                                    slc[ax] = slice(self.roi[r][0], self.roi[r][1])
                                    axes.append(ax)
                            else:
                                raise ValueError("roi must be 'horizontal' or 'vertical', found '{}'"
                                .format(str(r)))
                        # if the dimension is not in roi, average across the whole dimension
                        else:
                            ax = dataset.get_dimension_axis(r)
                            axes.append(ax)
                            self.roi.update({r:(0,dataset.get_dimension_size(r))})

                    self.flux = numpy.mean(dataset.array[tuple(slc)], axis=tuple(axes))

                else:
                    raise TypeError("roi must be a dictionary, found {}"
                    .format(str(type(self.roi))))
        else:
            if self.flux is None:
                raise ValueError("Please specify flux or roi")

        flux_size = (numpy.shape(self.flux))
        if len(flux_size) > 0:
            data_size = numpy.shape(dataset.geometry.angles)
            if data_size != flux_size:
                raise ValueError("Flux must be a scalar or array with length \
                                    \n = data.geometry.angles, found {} and {}"
                                    .format(flux_size, data_size))
            
        return True

    def show_roi(self, angle_index='range'):
        if angle_index=='range':
            data = self.get_input()
            max_zscore = numpy.argmax(numpy.abs(stats.zscore(self.flux)))
            self._plot_slice_roi_angle(angle_index=0, ax=231)
            self._plot_slice_roi_angle(angle_index=max_zscore, ax=232)
            self._plot_slice_roi_angle(angle_index=len(data.geometry.angles)-1, ax=233)

            plt.subplot(212)
            plt.plot(self.flux)
            plt.plot(max_zscore, self.flux[max_zscore],'rx')
        
            plt.legend(['Mean intensity in roi','Max Z-score'])
            plt.xlabel('Angle index')
            plt.ylabel('Mean intensity in roi')
            plt.tight_layout()

        elif isinstance(angle_index, int):
            self._plot_slice_roi_angle(angle_index=angle_index)

        else:
            raise TypeError("angle must be an int or 'range'")

    def _plot_slice_roi_angle(self, angle_index, ax=111):
        data = self.get_input()
        data_slice = data.get_slice(angle=angle_index)
        plt.subplot(ax)
        plt.imshow(data_slice.array, cmap='gray',aspect='equal', origin='lower')

        h = data_slice.dimension_labels[0]
        v = data_slice.dimension_labels[1]
        plt.plot([self.roi[v][0],self.roi[v][0]], [self.roi[h][0],self.roi[h][1]],'--r')
        plt.plot([self.roi[v][1],self.roi[v][1]], [self.roi[h][0],self.roi[h][1]],'--r')

        plt.plot([self.roi[v][0],self.roi[v][1]], [self.roi[h][0],self.roi[h][0]],'--r')
        plt.plot([self.roi[v][0],self.roi[v][1]], [self.roi[h][1],self.roi[h][1]],'--r')

        plt.xlabel(v)
        plt.ylabel(h)
        plt.title('Angle index = ' + str(angle_index))

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
