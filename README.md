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
 
 This package allows writing of optimisation algorithms. The main actors here are:
 
  * `Function`
  * `Operator`
 
 
 
