Developer's guide
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

It should not accept positional arguments `*args` or keyworded arguments `**kwargs` so that the user can clearly understand what parameters are necessary to 
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

Docstrings
==========


The Core Imaging Library follows the `NumpyDoc <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
style with the `PyData Sphinx html theme <https://pydata-sphinx-theme.readthedocs.io/en/latest/>`_.

When contributing your code please refer to `this <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ link 
for docstring formatting and this rendered `example <https://numpydoc.readthedocs.io/en/latest/example.html#example>`_ .


Contributions guidelines
========================

Make sure that each contributed file contains the following text enclosed in the appropriate comment syntax for the file format. Please replace `[yyyy]` and `[name of copyright owner]` with your own identifying information. Optionally you may add author name and email.

::

  Copyright 2022 United Kingdom Research and Innovation
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
