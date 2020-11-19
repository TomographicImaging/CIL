* 20.*
  - python module renamed to cil
  - renamed Identity->IdentityOperator, Gradient->GradientOperator, SymmetrisedGradient->SymmetrisedGradientOperator

* 20.09.1
  - FiniteDifferenceOperator takes into consideration voxel size
  - Added CentreOfRotationCorrector
  - Removed CenterOfRotationFinder
  - moved TestData to utilities and renamed as dataexample
  - verbosity of Algorithms is independent of the update_objective_interval
  - added unittests
  - renamed
    - GradientDescent to GD
    - SparseFiniteDiff to SparseFiniteDifferenceOperator
    - LinearOperatorMatrix to MatrixOperator
  - bugfix update_objective of SPDHG 

* 20.09
  - added SPDHG algorithm
  - added TotalVariation function
  - Redesign of the AcquisitionGeometry class allowing more general acquisition trajectories than currently possible.
  - Added ZEISS reader

* 20.04
  - Significant upgrades to the operator and function classes to allow more flexible definition of optimisation problems
  - Added multithreaded C library for calculation of finite difference and some data processing
  - Added Gradient operator using C library and numpy backends

* 19.10
  - Improved usability with reader/writers and plotting utilities
  - Substantially improved test coverage

* 19.07
  - Introduction of BlockFramework
  - major revision and restructuring of the whole code
  - rewritten io package

* 19.02
  - introduction of Algorithm class
  - unit test expanded and moved to test directory
  - unified build system on Jenkins based on CCPi-VirtualMachine repo
 - switched to calendar versioning YY.0M.

* 0.10.0

* 0.9.4
  - Initial release
