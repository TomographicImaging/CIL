Recon
*****

This module allows the user to run pre-configured reconstruction algorithms on their data.


Analytical Reconstruction
=========================

The CIL analytical reconstructions use CIL to filter and prepare the data using highly optimised routines. The filtered data is then
backprojected using projectors from TIGRE or ASTRA-TOOLBOX.

Standard FBP (filtered-backprojection) should be used for parallel-beam data. FDK (Feldkamp, Davis, and Kress) is a filtered-backprojection 
algorithm for reconstruction of cone-beam data measured with a standard circular orbit.

The filter can be set to a predefined function, or a custom filter can be set. The predefined filters take the following forms:

.. figure:: images/FBP_filters1.png
    :align: center
    :alt: FBP Filters
    :figclass: align-center


FBP - Reconstructor for parallel-beam geometry
----------------------------------------------


.. autoclass:: cil.recon.FBP
   :members:
   :inherited-members:


FDK - Reconstructor for cone-beam geometry
------------------------------------------


.. autoclass:: cil.recon.FDK
   :members:
   :inherited-members:


:ref:`Return Home <mastertoc>`