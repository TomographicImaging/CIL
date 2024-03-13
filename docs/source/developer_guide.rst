..    Copyright 2020 United Kingdom Research and Innovation
      Copyright 2020 The University of Manchester

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

Developers' Guide
*****************

CIL is an Object Orientated software. It has evolved during the years and it currently does not fully adheres to the following conventions. New additions must comply with
the following.

Conventions on new CIL objects
==============================

For each class there are **essential**, and **non-essential** parameters. The non-essential can be further be divided in **often configured** and **advanced** parameters:

* essential
* non-essential

  * often-configured
  * advanced

The definition of what are the essential, often-configured and advanced parameters depends on the actual class.

Creator
-------

To create an instance of a class, the creator of a class should require the **essential** and **often-configured** parameters as named parameters.

It should not accept positional arguments `*args` or key-worded arguments `**kwargs` so that the user can clearly understand what parameters are necessary to
create the instance.

Setter methods and properties
-----------------------------

Use of `property` is favoured instead of class members to store parameters so that the parameters can be protected.

The class should provide setter methods to change all the parameters at any time. Setter methods to set multiple parameters at the same time is also accepted.
Setter methods should be named `set_<parameter>`. The use of `set_` helps IDEs and user to find what they should change in an instance of a class.


Other methods
-------------

Methods that are not meant to be used by the user should have a `_` (underscore) at the beginning of the name.
All methods should follow the convention of small caps underscore separated words.

Documentation
=============

Docstrings
----------

The Core Imaging Library (CIL) follows the `NumpyDoc <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
style with the `PyData Sphinx HTML theme <https://pydata-sphinx-theme.readthedocs.io/en/latest/>`_.
When contributing your code please refer to `this <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ link
for docstring formatting and this rendered `example <https://numpydoc.readthedocs.io/en/latest/example.html#example>`_.

Example from ``cil``
^^^^^^^^^^^^^^^^^^^^

The following provides an example of the docstring format used within ``cil``, and the rendered documentation generated from it.

Source
""""""

.. literalinclude:: ../../Wrappers/Python/cil/recon/FBP.py
   :caption: `FBP.run method from cil.io.recon.FBP`
   :language: python
   :pyobject: FBP.run

Rendered
""""""""

.. automethod:: cil.recon.FBP.FBP.run


Building documentation locally
------------------------------

The easiest way to test changes to documentation is to build the docs locally. To do this, you will need a working ``cil`` installation.
For development of the documentation embedded within the source code itself (e.g. docstrings), you should build ``cil`` from source.

The following steps can be used to create an environment that is suitable for building ``cil`` and its documentation, and to start
a HTTP server to view the documentation.

#. Clone ``cil`` repo
#. Update conda with ``conda update -n base -c defaults conda``
#. Follow the instructions `here <https://github.com/TomographicImaging/CIL/tree/master#building-cil-from-source-code>`_ to create a conda environment and build ``cil`` from source
#. Go to ``docs`` folder
#. Install packages from ``docs/docs_environment.yml`` (with 'name' changed to ENVIRONMENT_NAME) using ``conda env update -f docs_environment.yml``
#. Download the notebooks for rendering in the documentation with ``python mkdemos.py``
#. Build the documentation ``sphinx-build -b dirhtml source build``
#. Start a HTTP server to serve documentation with ``python -m http.server --directory build``

Example:
::

  git clone --recurse-submodule git@github.com:TomographicImaging/CIL.git
  cd CIL
  sh scripts/create_local_env_for_cil_development_tests.sh -n NUMPY_VERSION -p PYTHON_VERSION -e ENVIRONMENT_NAME
  conda activate ENVIRONMENT_NAME
  cmake -S . -B ./build -DCMAKE_INSTALL_PREFIX=${CONDA_PREFIX}
  cmake --build ./build --target install
  cd docs
  conda update -n base -c defaults conda
  conda env update -f docs_environment.yml # with the name field set to ENVIRONMENT_NAME
  python mkdemos.py
  sphinx-build -b dirhtml source build
  python -m http.server -d build

Notebooks gallery
-----------------

The ``mkdemos.py`` script:

- downloads notebooks from external URLs to ``source/demos/*.ipynb``
- uses the ``demos-template.rst`` file to generate the gallery in ``source/demos.rst``

The ``nbsphinx`` extension will convert the ``*.ipynb`` files to HTML.

Contributions guidelines
========================

Make sure that each contributed file contains the following text enclosed in the appropriate comment syntax for the file format. Please replace `[yyyy]` and `[name of copyright owner]` with your own identifying information. Optionally you may add author name and email.

::

..Copyright 2022 United Kingdom Research and Innovation
  Copyright 2022 The University of Manchester
  Copyright [yyyy] [name of copyright owner]

  Author(s): [Author name, Author email (optional)]

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
