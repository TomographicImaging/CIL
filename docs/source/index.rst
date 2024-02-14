.. Copyright 2019 United Kingdom Research and Innovation
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

Welcome to CIL's documentation!
###############################

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
optimisation algorithms to solve generically formulated optimisation problems
constructed from operator and function objects.

Demos and Examples
==================
A number of demos can be found in the `CIL-Demos`_ repository.

For detailed information refer to our articles and the repositories
with the code to reproduce the article's results.

1. Jørgensen JS et al. 2021 Core Imaging Library Part I: a versatile python framework for tomographic imaging
https://doi.org/10.1098/rsta.2020.0192 . Phil. Trans. R. Soc. A 20200192.
The code to reproduce the article results. https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-I

2. Papoutsellis E et al. 2021 Core Imaging Library - Part II: multichannel reconstruction for dynamic and spectral
tomography https://doi.org/10.1098/rsta.2020.0193 Phil. Trans. R. Soc. A 20200193.
The code to reproduce the article results. https://github.com/TomographicImaging/Paper-2021-RSTA-CIL-Part-II


Cite this work
==============

If you use this software please consider citing one or both of the articles above.

Software documentation Index
****************************

.. toctree::
   :maxdepth: 3
   :caption: Contents:
   :name: mastertoc

   introduction
   framework
   io
   optimisation
   processors
   recon
   utilities
   plugins
   developer_guide
   demos

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`

Contacts
========

Please refer to the main `CCPi website`_ for up-to-date information.

The CCPi developers may be contacted:

* by joining the `devel mailing list`_
* on the CIL's GitHub repository page https://github.com/TomographicImaging/CIL or
* on the CIL Discord channel https://discord.gg/9NTWu9MEGq

.. _devel mailing list: https://www.jiscmail.ac.uk/cgi-bin/webadmin?A0=CCPI-DEVEL
.. _CCPi website: https://www.ccpi.ac.uk
.. _CIL-Demos: https://github.com/vais-ral/CIL-Demos
