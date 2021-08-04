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

    def __init__(
            self, data=None, file_name=None, flat_fields=None,
            dark_fields=None):
        '''
        data: Acqusistion Data ordered: (angle, vertical, horizontal)
        file_name: name of nexus file to write to
        flat_fields: numpy array of flat field data
        dark_fields: numpy array of dark field data
        '''

        self.data = data
        self.file_name = file_name
        self.flat_fields = flat_fields
        self.dark_fields = dark_fields

        if ((self.data is not None) and (self.file_name is not None)):
            self.set_up(data=self.data,
                        file_name=self.file_name, flat_fields=flat_fields,
                        dark_fields=dark_fields)

    def set_up(self, data=None, file_name=None,
               flat_fields=None, dark_fields=None):

        self.data = data
        self.file_name = file_name
        self.flat_fields = flat_fields
        self.dark_fields = dark_fields

        if not ((isinstance(self.data, ImageData)) or
                (isinstance(self.data, AcquisitionData))):
            raise Exception('Writer supports only following data types:\n' +
                            ' - ImageData\n - AcquisitionData')

        # check that h5py library is installed
        if h5pyAvailable is False:
            raise Exception('h5py is not available, cannot write NEXUS files.')

    def _initialise_nexus_file(self, f, data_len):
        ''' Creates empty data-structure for NXTomo file:'''
        height = self.data.as_array().shape[1]
        width = self.data.as_array().shape[2]

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
            'data', (data_len, height, width), np.uint32)
        # was dtype=np.uint16 in EPAC code, is float32 in NEXUSDataWriter
        # TODO: figure out what type we should write
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
        if (isinstance(self.data, ImageData)):
            raise Exception(
                "Can't write ImageData to NXTomo file, \
                only AcquisitionData is supported.")

        # if the folder does not exist, create the folder
        if not os.path.isdir(os.path.dirname(self.file_name)) and \
                os.path.dirname(self.file_name) != '':
            os.mkdir(os.path.dirname(self.file_name))

        if self.flat_fields is not None:
            if len(self.flat_fields.shape) == 2:
                flat_fields_len = 1
            else:
                flat_fields_len = self.flat_fields.shape[0]
        else:
            flat_fields_len = 0
        if self.dark_fields is not None:
            if len(self.dark_fields.shape) == 2:
                dark_fields_len = 1
            else:
                dark_fields_len = self.dark_fields.shape[0]
        else:
            dark_fields_len = 0

        data_len = self.data.as_array(
        ).shape[0] + flat_fields_len + dark_fields_len

        # create the file
        with h5py.File(self.file_name, 'w') as f:
            dataset, rotationset, imagekeyset = self._initialise_nexus_file(
                f, data_len)

            height = self.data.as_array().shape[1]
            width = self.data.as_array().shape[2]

            data_to_write = np.empty((data_len, height, width))
            for i in range(data_len):
                if i < flat_fields_len:
                    data_to_write[i] = self.flat_fields[i]
                elif i < (flat_fields_len + dark_fields_len):
                    data_to_write[i] = self.dark_fields[i-flat_fields_len]
                else:
                    data_to_write[i] = self.data.as_array(
                    )[i-dark_fields_len-flat_fields_len]

            dataset[...] = data_to_write

            # image key:
            data_len = self.data.as_array().shape[0]
            imagekey = [1] * flat_fields_len + [2] * \
                dark_fields_len + [0] * data_len
            imagekeyset[...] = imagekey

            # next is angles:
            rotation = self.data.geometry.config.angles.angle_data
            rotation = np.insert(
                rotation, 0, [rotation[0]] * (
                    flat_fields_len + dark_fields_len))
            rotationset[...] = rotation
