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

from cil.framework import AcquisitionGeometry
import numpy as np

h5pyAvailable = True
try:
    from h5py import File as NexusFile
except ImportError:
    h5pyAvailable = False


class NXTomoReader(object):
    '''
    Reader class for loading Nexus files in the NXTomo format:
    https://manual.nexusformat.org/classes/applications/NXtomo.html
    '''

    def __init__(self, file_name=None):
        '''
        This takes in input as filename and loads the data dataset.
        '''
        self.flat = None
        self.dark = None
        self.angles = None
        self.geometry = None
        self.filename = file_name
        self.key_path = 'entry1/tomo_entry/instrument/detector/image_key'
        self.data_path = 'entry1/tomo_entry/data/data'
        self.angle_path = 'entry1/tomo_entry/data/rotation_angle'

    def get_image_keys(self):
        '''
        Loads the image keys from the nexus file
        returns: numpy array containing image keys:
        0 = projection
        1 = flat field
        2 = dark field
        3 = invalid
        '''
        try:
            with NexusFile(self.filename, 'r') as file:
                return np.array(file[self.key_path])
        except KeyError as ke:
            raise KeyError("get_image_keys: ", ke.args[0], self.key_path)

    def load(self, dimensions=None, image_key_id=0):
        '''
        This is generic loading function of flat field, dark field and
        projection data.
        Loads data with image key id of image_key_id
        dimensions: a tuple of 'slice'
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:
            with NexusFile(self.filename, 'r') as file:
                image_keys = np.array(file[self.key_path])
                projections = None
                if dimensions is None:
                    projections = np.array(file[self.data_path])
                    result = projections[image_keys == image_key_id]
                    return result
                else:
                    # When dimensions are specified they need to be mapped to
                    # image_keys
                    index_array = np.where(image_keys == image_key_id)
                    projection_indexes = index_array[0][dimensions[0]]
                    new_dimensions = list(dimensions)
                    new_dimensions[0] = projection_indexes
                    new_dimensions = tuple(new_dimensions)
                    result = np.array(file[self.data_path][new_dimensions])
                    return result
        except Exception:
            print("Error reading nexus file")
            raise

    def load_projection(self, dimensions=None):
        '''
        Loads the projection data from the nexus file.
        dimensions: a tuple of 'slice's
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        returns: numpy array with projection data
        '''
        try:
            if 0 not in self.get_image_keys():
                raise ValueError("Projections are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 0)

    def load_flat(self, dimensions=None):
        '''
        Loads the flat field data from the nexus file.
        dimensions: a tuple of 'slice's
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        returns: numpy array with flat field data
        '''
        try:
            if 1 not in self.get_image_keys():
                raise ValueError("Flats are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 1)

    def load_dark(self, dimensions=None):
        '''
        Loads the Dark field data from the nexus file.
        dimensions: a tuple of 'slice's
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        returns: numpy array with dark field data
        '''
        try:
            if 2 not in self.get_image_keys():
                raise ValueError("Darks are not in the data. Data Path ",
                                 self.data_path)
        except KeyError as ke:
            raise KeyError(ke.args[0], self.data_path)
        return self.load(dimensions, 2)

    def get_projection_angles(self):
        '''
        Loads the projection angles from the nexus file.
        returns: array containing the projection angles
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:
            with NexusFile(self.filename, 'r') as file:
                angles = np.array(file[self.angle_path], np.float32)
                image_keys = np.array(file[self.key_path])
                return angles[image_keys == 0]
        except Exception:
            print("get_projection_angles Error reading nexus file")
            raise

    def get_sinogram_dimensions(self):
        '''
        Return the sinogram dimensions of the dataset
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:
            with NexusFile(self.filename, 'r') as file:
                projections = file[self.data_path]
                image_keys = np.array(file[self.key_path])
                dims = list(projections.shape)
                dims[0] = dims[1]
                dims[1] = np.sum(image_keys == 0)
                return tuple(dims)
        except Exception:
            print("Error reading nexus file")
            raise

    def get_projection_dimensions(self):
        '''
        Return the projection dimensions of the dataset
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:
            with NexusFile(self.filename, 'r') as file:
                try:
                    projections = file[self.data_path]
                except KeyError as ke:
                    raise KeyError('Error: data path {0} not found\n{1}'
                                   .format(self.data_path,
                                           ke.args[0]))
                image_keys = self.get_image_keys()
                dims = list(projections.shape)
                dims[0] = np.sum(image_keys == 0)
                return tuple(dims)
        except Exception:
            print(
                "Warning: Error reading image_keys trying accessing data on ",
                self.data_path)
            with NexusFile(self.filename, 'r') as file:
                dims = file[self.data_path].shape
                return tuple(dims)

    def get_acquisition_data(self, dimensions=None):
        '''
        Loads the acquisition data within the given dimensions and returns
        an AcquisitionData Object
        dimensions: a tuple of 'slice'
        e.g. (slice(0, 1), slice(0, 135), slice(0, 160))
        '''
        data = self.load_projection(dimensions)
        dims = self.get_projection_dimensions()
        geometry = AcquisitionGeometry.create_Parallel3D().set_panel(
            num_pixels=(dims[2], dims[1]), pixel_size=(1, 1)).set_angles(
            angles=self.get_projection_angles()).set_labels(
            ['angle', 'vertical', 'horizontal']).set_channels(1)
        out = geometry.allocate()
        out.fill(data)
        return out

    def get_acquisition_data_subset(self, ymin=None, ymax=None):
        '''
        This method load the acquisition data, cropped in the vertical
        direction, from ymin to ymax, and returns
        an AcquisitionData object
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:

            with NexusFile(self.filename, 'r') as file:
                try:
                    dims = self.get_projection_dimensions()
                except KeyError:
                    pass
                dims = file[self.data_path].shape
                if ymin is None and ymax is None:

                    try:
                        image_keys = self.get_image_keys()
                        projections = np.array(file[self.data_path])
                        data = projections[image_keys == 0]
                    except KeyError as ke:
                        print(ke)
                        data = np.array(file[self.data_path])

                else:
                    image_keys = self.get_image_keys()
                    projections = np.array(file[self.data_path])[
                        image_keys == 0]
                    if ymin is None:
                        ymin = 0
                        if ymax > dims[1]:
                            raise ValueError('ymax out of range')
                        data = projections[:, :ymax, :]
                    elif ymax is None:
                        ymax = dims[1]
                        if ymin < 0:
                            raise ValueError('ymin out of range')
                        data = projections[:, ymin:, :]
                    else:
                        if ymax > dims[1]:
                            raise ValueError('ymax out of range')
                        if ymin < 0:
                            raise ValueError('ymin out of range')

                        data = projections[:, ymin:ymax, :]

        except Exception:
            print("Error reading nexus file")
            raise

        try:
            angles = self.get_projection_angles()
        except KeyError:
            n = data.shape[0]
            angles = np.linspace(0, n, n+1, dtype=np.float32)

        if ymax-ymin > 1:
            geometry = AcquisitionGeometry.create_Parallel3D().set_panel(
                num_pixels=(dims[2], ymax-ymin), pixel_size=(1, 1)).set_angles(
                angles=angles).set_labels(
                    ['angle', 'vertical', 'horizontal']).set_channels(1)
            out = geometry.allocate()
            out.fill(data)
            return out
        elif ymax-ymin == 1:
            geometry = AcquisitionGeometry.create_Parallel2D().set_panel(
                num_pixels=(dims[2]), pixel_size=(1, 1)).set_angles(
                angles=angles).set_labels(
                    ['angle', 'horizontal']).set_channels(1)
            out = geometry.allocate()
            out.fill(data.squeeze())
            return out

    def get_acquisition_data_slice(self, y_slice=0):
        ''' Returns a vertical slice of the projection data at y_slice,
         as an AcquisitionData object'''
        return self.get_acquisition_data_subset(ymin=y_slice, ymax=y_slice+1)

    def get_acquisition_data_whole(self):
        # TODO: Is this necessary?
        '''
        Loads the acquisition data and returns an AcquisitionData Object
        Same behaviour as get_acquisition_data when dimensions is set to None
        '''
        with NexusFile(self.filename, 'r') as file:
            try:
                dims = self.get_projection_dimensions()
            except KeyError as ke:
                print("Warning: ", ke)
                dims = file[self.data_path].shape

            ymin = 0
            ymax = dims[1]

            return self.get_acquisition_data_subset(ymin=ymin, ymax=ymax)

    def list_file_content(self):
        '''
        Prints paths to all datasets within nexus file.
        '''
        try:
            with NexusFile(self.filename, 'r') as file:
                file.visit(print)
        except Exception:
            print("Error reading nexus file")
            raise

    def get_acquisition_data_batch(self, bmin=None, bmax=None):
        # TODO: Perhaps we should rename?
        '''
        This method load the acquisition data, cropped in the angular
        direction, from index bmin to bmax, and returns
        an AcquisitionData object
        '''
        if not h5pyAvailable:
            raise Exception("Error: h5py is not installed")
        if self.filename is None:
            return
        try:

            with NexusFile(self.filename, 'r') as file:
                dims = self.get_projection_dimensions()
                if bmin is None or bmax is None:
                    raise ValueError(
                        'get_acquisition_data_batch: please specify fastest \
                        index batch limits')

                if bmin >= 0 and bmin < bmax and bmax <= dims[0]:
                    image_keys = np.array(file[self.key_path])
                    projections = None
                    projections = np.array(file[self.data_path])
                    data = projections[image_keys == 0]
                    data = data[bmin:bmax]
                else:
                    raise ValueError('get_acquisition_data_batch: bmin {0}>0 bmax {1}<{2}'.format(
                        bmin, bmax, dims[0]))

        except Exception:
            print("Error reading nexus file")
            raise

        try:
            angles = self.get_projection_angles()[bmin:bmax]
        except KeyError:
            n = data.shape[0]
            angles = np.linspace(0, n, n+1, dtype=np.float32)[bmin:bmax]

        if bmax-bmin >= 1:

            geometry = AcquisitionGeometry('parallel', '3D',
                                           angles=angles,
                                           pixel_num_h=dims[2],
                                           pixel_size_h=1,
                                           pixel_num_v=dims[1],
                                           pixel_size_v=1,
                                           dist_source_center=None,
                                           dist_center_detector=None,
                                           channels=1,
                                           dimension_labels=[
                                               'angle', 'vertical',
                                               'horizontal'])
            out = geometry.allocate()
            if bmax-bmin == 1:
                out.fill(data.squeeze())
            else:
                out.fill(data)
            return out
