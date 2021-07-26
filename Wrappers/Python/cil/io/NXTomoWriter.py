# -*- coding: utf-8 -*-
#   This work is part of the Core Imaging Library (CIL) developed by CCPi
#   (Collaborative Computational Project in Tomographic Imaging), with
#   substantial contributions by UKRI-STFC and University of Manchester.

#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at

#   http://www.apache.org/licenses/LICENSE-2.0

#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import numpy as np
import os
from cil.framework import AcquisitionData, ImageData


h5pyAvailable = True
try:
    import h5py
except ImportError:
    h5pyAvailable = False


class NXTomoWriter(object):

    def __init__(self, data=None, filename=None, flat_field=[], dark_field=[]):

        self.data = data
        self.file_name = filename
        self.flat_field = flat_field
        self.dark_field = dark_field

        if ((self.data is not None) and (self.file_name is not None)):
            self.set_up(data=self.data,
                        file_name=self.file_name, flat_field=flat_field,
                        dark_field=dark_field)

    def set_up(self,
               data=None,
               file_name=None, flat_field=None, dark_field=None):

        self.data = data
        self.file_name = file_name

        if not ((isinstance(self.data, ImageData)) or
                (isinstance(self.data, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        # check that h5py library is installed
        if h5pyAvailable is False:
            raise Exception('h5py is not available, cannot write NEXUS files.')

    def initialise_nexus_file(self, f):
        data_len = self.data.as_array(
        ).shape[0] + self.flat_field.shape[0] + self.dark_field.shape[0]
        height = self.data.as_array().shape[1]
        width = self.data.as_array().shape[2]  # TODO: check ordering

        # Create empty data-structure for NXTomo file:
        entry = f.create_group('entry1')
        entry.attrs['NX_class'] = 'NXentry'

        entry = entry.create_group('tomo_entry')
        entry.attrs['NX_class'] = 'NXsubentry'
        entry['definition'] = 'NXtomo'

        instrument = entry.create_group('instrument')
        instrument.attrs['NX_class'] = 'NXinstrument'

        detector = instrument.create_group('detector')
        detector.attrs['NX_class'] = 'NXdetector'

        sample = entry.create_group('sample')
        sample.attrs['NX_class'] = 'NXsample'
        sample['name'] = 'anonymous sample'

        data = entry.create_group('data')
        data.attrs['NX_class'] = 'NXdata'

        dataset = detector.create_dataset(
            'data', (data_len, height, width), np.float32)  # was dtype=np.uint16)
        dataset.attrs['long_name'] = 'X-ray counts'

        imagekeyset = detector.create_dataset(
            'image_key', (data_len,), dtype=np.uint8)

        rotationset = sample.create_dataset(
            'rotation_angle', (data_len,), dtype=np.float32)
        rotationset.attrs['units'] = 'degrees'
        rotationset.attrs['long_name'] = 'Rotation angle'

        data['data'] = dataset
        dataset.attrs['target'] = dataset.name
        data['rotation_angle'] = rotationset
        rotationset.attrs['target'] = rotationset.name
        data['image_key'] = imagekeyset
        imagekeyset.attrs['target'] = imagekeyset.name

        data.attrs['signal'] = 'data'
        data.attrs.create('axes', ['rotation_angle', '.', '.'],
                          dtype=h5py.special_dtype(vlen=str))
        data.attrs['rotation_angle_indices'] = 0

        # Now add in CIL identifier that we have AcquisitionData:
        data.attrs['data_type'] = 'AcquisitionData'

        return dataset, rotationset, imagekeyset

    def write(self):
        # if the folder does not exist, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)):
            os.mkdir(os.path.dirname(self.file_name))

        # create the file
        with h5py.File(self.file_name, 'w') as f:
            dataset, rotationset, imagekeyset = self.initialise_nexus_file(f)
            print(type(dataset), type(rotationset), type(imagekeyset))
            # set up dataset attributes
            if (isinstance(self.data, ImageData)):
                print("Can't write ImageData to NXTomo file")

            # set up dataset attributes
            data_len = self.data.as_array(
            ).shape[0] + self.flat_field.shape[0] + self.dark_field.shape[0]
            height = self.data.as_array().shape[1]
            width = self.data.as_array().shape[2]
            data_to_write = self.data.as_array()

            data_to_write = np.empty((data_len, height, width))
            for i in range(data_len):
                if i < self.flat_field.shape[0]:
                    data_to_write[i] = self.flat_field[i]
                elif i < (self.flat_field.shape[0] + self.dark_field.shape[0]):
                    data_to_write[i] = self.dark_field[
                        i - self.flat_field.shape[0]]
                else:
                    data_to_write[i] = self.data.as_array(
                    )[i-self.dark_field.shape[0]-self.flat_field.shape[0]]

            dataset[...] = data_to_write

            # image key:
            data_len = self.data.as_array().shape[0]
            flat_field_len = self.flat_field.shape[0]
            dark_field_len = self.dark_field.shape[0]
            imagekey = [1] * flat_field_len + [2] * \
                dark_field_len + [0] * data_len
            imagekeyset[...] = imagekey

            # next is angles:
            rotation = self.data.geometry.config.angles.angle_data
            rotation = np.insert(
                rotation, 0, [rotation[0]] * (flat_field_len + dark_field_len))
            rotationset[...] = rotation
