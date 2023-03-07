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
  writer.set_up(file_name='tmp_nexus.nxs',
              data_container=my_data)
  # write data
  writer.write_file()

  # read data
  # initialize NEXUS reader
  reader = NEXUSDataReader()
  reader.set_up(nexus_file='tmp_nexus.nxs')
  # load data
  ad1 = reader.load_data()
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
