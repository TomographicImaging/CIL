# CCPi-PythonFramework
Basic Python Framework for CIL

This package aims at ensuring a longer life and easy extensibility of the CIL software. This package provides a common framework, hence the name, for the analysis of data in the CT pipeline and quick development of novel reconstruction algorithms.

Some concepts are so much overlapping with the CCPPETMR project that we have chosen to stick to their naming and conventions, in the hope that we may be able to complement each other (at least in Python).

### Components

This package consists of the following Python modules:
1. `ccpi.framework`
2. `ccpi.optimisation`

### `ccpi.framework`

In `ccpi.framework` we define a number of common classes normally used in tomography:
 
 * `DataContainer`
 * `DataSetProcessor`
 * `ImageData`
 * `AcquisitionData`
 
 #### `DataContainer`
 Generic class to hold data. Currently the data is currently held in a numpy arrays, but we are currently planning to create a `GPUDataContainer` and `BoostDataContainer` which will hold the data in an array on GPU or in a boost multidimensional array respectively. 
 
 The role of the `DataContainer` is to hold the data and metadata as axis labels and geometry.
 `ImageData` and `AcquisitionData` are subclasses aimed at containing 3D/4D data and raw data.
 
 `DataContainer` have basic methods to perform algebric operations between each other and/or with numbers. `DataContainer` provide a simple method to produce a `subset` of themselves based on the axis one would like to have. For instance if a `DataContainer` `A` is 3D and its axis represent temperature, width, height, one could create a reordered `DataContainer` by 
 ```python
 
 B = A.subset(['height', 'width','temperature'])
 C = A.subset(temperature = 20)
 ```
 
 #### `DataSetProcessor`
 Defines a generic DataContainer processor, it accepts `DataContainer` as inputs and outputs `DataContainer`.
 The aim of this class is to simplify the writing of processing pipelines. 
 
 A `DataSetProcessor` does calculate its output only when needed and can return a copy of its output (if available) when none of its inputs have changed. Normally it is important to overwrite the `process` method and the `__init__` to describe all the parameter that are specific to the processor.
 
 ### `ccpi.optimisation`
 
 This package allows rapid prototyping of optimisation-based reconstruction problems, 
 i.e. defining and solving different optimization problems to enforce different properties 
 on the reconstructed image.
 
 Firstly, it provides an object-oriented framework for defining mathematical operators and functions 
 as well a collection of useful example operators and functions. Both smooth and 
 non-smooth functions can be used. 
 
 Further, it provides a number of high-level generic 
 implementations of optimisation algorithms to solve genericlly formulated 
 optimisation problems constructed from operator and function objects. 
 
 The fundamental components are:
 
  * `Operator`: A class specifying a (currently linear) operator
  * `Function`: A class specifying mathematical functions such as a least squares data fidelity.
  * `Algorithm`: Implementation of an optimisation algorithm to solve a particular generic optimisation problem. These are currently python functions by may be changed to operators in another release.

 #### `Operator`
 
 The two most important methods are `direct` and `adjoint` methods that describe the result of 
 applying the operator, and its adjoint respectively, onto a compatible `DataContainer` input. 
 The output is another `DataContainer` object or subclass hereof. An important 
 special case is to represent the tomographic forward and backprojection operations.
 
 #### `Function`
 
 A `function` represents a mathematical function of one or more inputs is intended 
 to accept `DataContainer`s as input as well as any additional parameters. 
 Its methods reflect the properties of the function, for example, 
 if the function represented is differentiable 
 the `function` should contain a method `grad` which should return the gradient of the function evaluated at
 an input point. If the function is not differentiable but allows a simple proximal operator, the method 
 `prox` should return the proxial operator evaluated at an input point. It is also possible 
 to evaluate the function value using the method `fun`.
 
 #### `Algorithm`
 
 A number of generic algorithm implementations are provided including CGLS and FISTA. An algorithm 
 is designed for a particular generic optimisation problem accepts and number of `function`s and/or 
 `operator`s as input to define a specific instance of the generic optimisation problem to be solved.
 
 #### Examples
 
 Please see the demos for examples of defining and using operators, functions and algorithms 
 to specify and solve optimisation-based reconstruction problems.
 
 
 
