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
   :special-members:
.. autoclass:: cil.io.NEXUSDataWriter
   :members:
   :special-members:
|

Nikon
=====
.. autoclass:: cil.io.NikonDataReader
   :members:
   :special-members:

ZEISS
=====

.. autoclass:: cil.io.TXRMDataReader
   :members:
   :special-members:

TIFF Reader/Writer
==================

.. autoclass:: cil.io.TIFFStackReader
   :members:
   :special-members:

.. autoclass:: cil.io.TIFFWriter
   :members:
   :special-members:

:ref:`Return Home <mastertoc>`
