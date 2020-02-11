Read/ write AcquisitionData and ImageData
*****************************************


NeXus
=====

The CCPi Framework provides classes to read and write :code:`AcquisitionData` and :code:`ImageData`
as NeXuS files.

.. code:: python

  # imports
  from ccpi.io import NEXUSDataWriter, NEXUSDataReader

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
  # get AcquisiionGeometry
  ag1 = reader.get_geometry()

.. autoclass:: ccpi.io.NEXUSDataReader
   :members:
   :special-members:
.. autoclass:: ccpi.io.NEXUSDataWriter
   :members:
   :special-members:
|

Nikon
=====
.. autoclass:: ccpi.io.NikonDataReader
   :members:
   :special-members:
|


:ref:`Return Home <mastertoc>`
