.. CCPi-Framework documentation master file, created by
   sphinx-quickstart on Tue Mar 19 15:12:44 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CCPi-Framework's documentation!
==========================================

The aim of this package is to enable rapid prototyping of optimisation-based 
reconstruction problems, i.e. defining and solving different optimization problems to enforce different properties on the reconstructed image, while being
powerful enough to be employed on real scale problems. 

Firstly, it provides a framework to handle acquisition and reconstruction
data and metadata; it also provides a basic input/output package to read data 
from different sources, e.g. Nikon X-Radia, NeXus.

Secondly, it provides an object-oriented framework for defining mathematical 
operators and functions as well a collection of useful example operators and 
functions. Both smooth and non-smooth functions can be used.

Further, it provides a number of high-level generic implementations of 
optimisation algorithms to solve genericlly formulated optimisation problems 
constructed from operator and function objects.

A number of demos can be found on the `CIL-Demos`_ repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :name: mastertoc


   framework
   io
   optimisation
   plugins
   astra
   contrib
   developer_guide

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

Contacts
========

Please refer to the main `CCPi website`_ for up-to-date information.

The CCPi developers may be contacted joining the `devel mailing list`_

.. _devel mailing list: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=CCPI-DEVEL
.. _CCPi website: https://www.ccpi.ac.uk
.. _CIL-Demos: https://github.com/vais-ral/CIL-Demos
