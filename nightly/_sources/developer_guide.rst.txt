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
All methods should follow the convention of small caps underscore separated words.Docstrings
----------


The Core Imaging Library follows the `NumpyDoc <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_
style with the `PyData Sphinx html theme <https://pydata-sphinx-theme.readthedocs.io/en/latest/>`_.

When contributing your code please refer to `this <https://numpydoc.readthedocs.io/en/latest/format.html#docstring-standard>`_ link 
for docstring formatting and this rendered `example <https://numpydoc.readthedocs.io/en/latest/example.html#example>`_ .




