..    Copyright 2019 United Kingdom Research and Innovation
      Copyright 2019 The University of Manchester

      Licensed under the Apache License, Version 2.0 (the "License");
      you may not use this file except in compliance with the License.
      You may obtain a copy of the License at

          http://www.apache.org/licenses/LICENSE-2.0

      Unless required by applicable law or agreed to in writing, software
      distributed under the License is distributed on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
      See the License for the specific language governing permissions and
      limitations under the License.

     Authors:
     CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
     Kyle Pidgeon (UKRI-STFC)

Read/ write AcquisitionData and ImageData
*****************************************


NeXuS
=====

The CCPi Framework provides classes to read and write :code:`AcquisitionData` and :code:`ImageData`
as NeXuS files.

.. code:: python

  # imports
  from cil.io import NEXUSDataWriter, NEXUSDataReader

  # initialise NEXUS Writer
  writer = NEXUSDataWriter()
  writer.set_up(data=my_data,
              file_name='tmp_nexus.nxs')
  # write data
  writer.write()

  # read data
  # initialize NEXUS reader
  reader = NEXUSDataReader()
  reader.set_up(file_name='tmp_nexus.nxs')
  # load data
  ad1 = reader.read()
  # get AcquisitionGeometry
  ag1 = reader.get_geometry()

.. autoclass:: cil.io.NEXUSDataReader
   :members:
   :inherited-members:
.. autoclass:: cil.io.NEXUSDataWriter
   :members:
   :inherited-members:
|

Nikon
=====
.. autoclass:: cil.io.NikonDataReader
   :members:
   :inherited-members:

ZEISS
=====
.. autoclass:: cil.io.ZEISSDataReader
   :members:
   :inherited-members:

TIFF Reader/Writer
==================

.. autoclass:: cil.io.TIFFStackReader
   :members:
   :exclude-members: set_up

.. autoclass:: cil.io.TIFFWriter
   :members:
   :exclude-members: set_up

RAW File Writer
===============

.. autoclass:: cil.io.RAWFileWriter
   :members:

:ref:`Return Home <mastertoc>`

HDF5 Utilities
==================

Utility functions to browse HDF5 files. These allow you to browse groups and read in datasets as numpy.ndarrays.

A CIL geometry and dataset must be constructed manually from the array and metadata.

.. autoclass:: cil.io.utilities.HDF5_utilities
   :members:


:ref:`Return Home <mastertoc>`
