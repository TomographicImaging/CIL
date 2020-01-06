
| Master | Development | Anaconda binaries |
|--------|-------------|-------------------|
| [![Build Status](https://anvil.softeng-support.ac.uk/jenkins/buildStatus/icon?job=CILsingle/CCPi-Framework)](https://anvil.softeng-support.ac.uk/jenkins/job/CILsingle/job/CCPi-Framework/) | [![Build Status](https://anvil.softeng-support.ac.uk/jenkins/buildStatus/icon?job=CILsingle/CCPi-Framework-dev)](https://anvil.softeng-support.ac.uk/jenkins/job/CILsingle/job/CCPi-Framework-dev/) |![conda version](https://anaconda.org/ccpi/ccpi-framework/badges/version.svg) ![conda last release](https://anaconda.org/ccpi/ccpi-framework/badges/latest_release_date.svg) [![conda platforms](https://anaconda.org/ccpi/ccpi-framework/badges/platforms.svg) ![conda dowloads](https://anaconda.org/ccpi/ccpi-framework/badges/downloads.svg)](https://anaconda.org/ccpi/ccpi-framework) |

# CCPi-Framework

Basic Python Framework for CIL

This package provides a common framework, hence the name, for the analysis of data in the CT pipeline and quick development of novel reconstruction algorithms.

## Installation

Binary installation of the CCPi Framework can be done with `conda`. Install a new environment as in the following. Some packages are optional (`cvxpy`, `ccpi-plugins`, `ccpi-astra`) but very useful.

```bash

conda create -y --name cil python=3 cvxpy ccpi-framework=19.10 ccpi-astra=19.10 tomophantom ccpi-plugins=19.10 -c oxfordcontrol -c conda-forge -c astra-toolbox -c ccpi
```

### Components

Some concepts are so much overlapping with the CCPPETMR [SIRF](https://github.com/CCPPETMR/SIRF) project that we have chosen to stick to their naming and conventions, in the hope that we may be able to complement each other (at least in Python).

This package consists of the following Python modules:
1. `ccpi.framework`
2. `ccpi.optimisation`
3. `ccpi.io`

### `ccpi.framework`

In `ccpi.framework` we define a number of common classes normally used in tomography:
 
 * `DataContainer`
 * `DataSetProcessor`
 * `ImageData`
 * `AcquisitionData`
 * `BlockDataContainer`
 
 #### `DataContainer`
 Generic class to hold data. Currently the data is currently held in a numpy arrays, but we are currently planning to create a `GPUDataContainer` and `BoostDataContainer` which will hold the data in an array on GPU or in a boost multidimensional array respectively. 
 
 The role of the `DataContainer` is to hold the data and metadata as axis labels and geometry.
 `ImageData` and `AcquisitionData` are subclasses aimed at containing 3D/4D data and raw data.
 
 `DataContainer` have basic methods to perform algebric operations between each other and/or with numbers. `DataContainer` provide a simple method to produce a `subset` of themselves based on the axis one would like to have. For instance if a `DataContainer` `A` is 3D and its axis represent temperature, width, height, one could create a reordered `DataContainer` by 
 ```python
 
 B = A.subset(['height', 'width','temperature'])
 C = A.subset(temperature = 20)
 ```
 
 #### `DataProcessor`
 Defines a generic DataContainer processor, it accepts `DataContainer` as inputs and outputs `DataContainer`.
 The aim of this class is to simplify the writing of processing pipelines. 
 
 A `DataProcessor` does calculate its output only when needed and can return a copy of its output (if available) when none of its inputs have changed. Normally it is important to overwrite the `process` method and the `__init__` to describe all the parameter that are specific to the processor.
 
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
 
  * `Operator`: A class specifying a operator
  * `Function`: A class specifying mathematical functions such as a least squares data fidelity.
  * `Algorithm`: Implementation of an iterative optimisation algorithm to solve a particular generic optimisation problem. Algorithms are iterable Python object which can be run in a for loop. Can be stopped and warm restarted. 

 #### `Operator`
 
 The two most important methods are `direct` and `adjoint` methods that describe the result of 
 applying the operator, and its adjoint respectively, onto a compatible `DataContainer` input. 
 The output is another `DataContainer` object or subclass hereof. An important 
 special case is to represent the tomographic forward and backprojection operations.
 
 #### `Function`
 
 A `function` represents a mathematical function of one or more inputs and is intended 
 to accept `DataContainer`s as input as well as any additional parameters. 
 Fixed parameters can be passed in during the creation of the `function` object.
 The methods of the `function` reflect the properties of it, for example, 
 if the function represented is differentiable 
 the `function` should contain a method `gradient` which should return the gradient of the function evaluated at
 an input point. If the function is not differentiable but allows a simple proximal operator, the method 
 `proximal` should return the proxial operator evaluated at an input point. The function value 
 is evaluated by calling the function itself, e.g. `f(x)` for a `function` 
 `f` and input point `x`.
 
 #### `Algorithm`
 
 A number of generic algorithm implementations are provided including Gradient Descent CGLS and FISTA. An algorithm 
 is designed for a particular generic optimisation problem accepts and number of `Function`s and/or 
 `Operator`s as input to define a specific instance of the generic optimisation problem to be solved.
 
 They are iterable objects which can be run in a `for` loop. The user can provide a stopping criterion different than the default max_iteration. `Algorithm`s provide a courtesy method `run(number_of_iterations, verbose)` which allows the user to easily run a `number_of_iterations` and receive a print to screen.
 
 New algorithms can be easily created by extending the `Algorithm` class. The user is required to implement only 4 methods: `set_up`, `__init__`, `update` and `update_objective`.
 
   * `set_up` and `__init__` are used to configure the algorithm
   * `update` is the actual iteration updating the solution
   * `update_objective` defines how the objective is calculated. 
 
 For example, the implementation of the `update` of the Gradient Descent algorithm to minimise a `Function` will only be:
 ```python
 def update(self):
    self.x += -self.rate * self.objective_function.gradient(self.x)
 def update_objective(self):
        self.loss.append(self.objective_function(self.x))
 ```
 
 The `Algorithm` provides the infrastructure to continue iteration, to access the values of the objective function in subsequent iterations, the time for each iteration. 
 
 ### Block Framework

Block Framework is a generic strategy to treat variational problems in the following form: `\min \text{Regulariser} + \text{Fidelity} `
where Regulariser and Fidelity are convex functions. While Fidelity is differentiable Regulariser may not.

We have now a number of algorithms that can address this:

 * GradientDescent
 * CGLS
 * SIRT
 * FISTA
 * PDHG

 #### Examples
 
 Please see the [demos](https://github.com/vais-ral/CIL-Demos) for examples of defining and using operators, functions and algorithms 
 to specify and solve optimisation-based reconstruction problems.
 
