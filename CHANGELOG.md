
* XX.X.X
  - Added wavelet operator, wrapping PyWavelets operator as a CIL operator
  - Added L1Sparsity function, allowing calculations of `|Ax-b|_1` and it's proximal, in the case of orthogonal operators, A
  - Bug fix in gradient descent `update_objective` called twice on the initial point.
  - ProjectionMap operator bug fix in adjoint and added documentation #1743
  - Our operators and functions now also return when out is specified #1742


* 24.0.0
  - Update to new CCPi-Regularisation toolkit v24.0.0. This is a backward incompatible release of the toolkit.
  - CIL plugin support for TIGRE version v2.6
  - CIL plugin support for ASTRA-TOOLBOX version v2.1
  - Dropped support for python 3.8 and 3.9
  - Added support for python 3.11 and 3.12
  - Dropped support for numpy 1.21 and 1.22
  - Added support for numpy 1.25 and 1.26
  - Set CMake Policy CMP0148 to OLD to avoid warnings in CMake 3.27
  - AcquisitionGeometry prints the first and last 10 angles, or all if there are 30 or less, rather than the first 20
  - Added a weight argument to the L1Norm function
  - Allow reduction methods on the DataContainer class to accept axis argument as string which matches values in dimension_labels
  - Added the functions `set_norms` and `get_norms` to the `BlockOperator` class
  - Internal variable name change in BlockOperator to aid understanding
  - Fixed formatting errors in the L2NormSquared and LeastSquares documentation
  - Bug fix for BlockDataContainer as iterator
  - Dropped support for IPP versions older than 2021.10 due to header changes
  - Fix build include directories
  - Proximal of MixedL21Norm with numpy backend now accepts numpy ndarray, DataContainer and float as tau parameter
  - ZeroOperator no longer relies on the default of allocate
  - Bug fix in SIRF TotalVariation unit tests with warm_start
  - Allow show2D to be used with 3D `DataContainer` instances
  - Added the a `Sampler` class as a CIL optimisation utility
  - Update documentation for symmetrised gradient
  - Added documentation for CompositionOperator and SumOperator
  - Updated FISTA and ISTA algorithms to allow input functions to be None
  - Bug fix in the adjoint of the Diagonal Operator for complex values
  - Update conda build action to v2 for 2.5x quicker builds
  - Add docker image, test & push to [`ghcr.io/tomographicimaging/cil`](https://github.com/TomographicImaging/CIL/pkgs/container/cil)
  - Quality metrics have added mask option
  - Bug fix for norm of CompositionOperator
  - Functions updated to use sapyb
  - Fix KullbackLeibler numba gradient ignoring mask
  - show1D slice_list parameter can now be of type tuple
  - Deprecated `Algorithm`'s `max_iteration`, `log_file`, `**kwargs`, `max_iteration_stop_criterion`, `objective_to_string`, `verbose_output`, `verbose_header`, `run(print_interval)`
  - Added `optimisation.utilities.callbacks`
    - Added `Callback` (abstract base class), `ProgressCallback`, `TextProgressCallback`, `LogfileCallback`
    - Deprecated `Algorithm.run(callback: Callable)`
    - Added `Algorithm.run(callbacks: list[Callback])`
  - Allow `CentreOfRotationCorrector.xcorrelation` to accept a list of two values in `projection_index` to use for the correlation
  - Bug fix in `CentreOfRotationCorrector.xcorrelation` finding correlation angles for limited angle data
  - New unit tests have been implemented for operators and functions to check for in place errors and the behaviour of `out`.
  - Bug fix for missing factor of 1/2 in SIRT update objective and catch in place errors in the SIRT constraint
  - Bug fix to allow safe in place calculation for the soft shrinkage algorithm
  - Allow Masker to take integer arrays in addition to boolean
  - Add remote data class to example data to enable download of relevant datasets from remote repositories 
  - Improved import error/warning messages
  - New adjoint operator
  - Bug fix for complex matrix adjoint
  - Removed the following code which was deprecated since v22.0.0:
    - `info` parameter in `cil.optimisation.functions.TotalVariation`
    - `sinogram_geometry` and `volume_geometry` parameters in `cil.plugins.astra.processors.FBP` and in `cil.plugins.tigre.processors.FBP`
    - `aquisition_geometry` (misspelled) parameter in `cil.plugins.tigre.processors.ProjectionOperator`
    - `FBP` kwarg (and thus all kwargs) in `cil.processors.CentreOfRotationCorrector` and `cil.processors.CofR_image_sharpness`
    - `TXRMDataReader`
  - Added the ApproximateGradientSumFunction and SGFunction to allow for stochastic gradient algorithms to be created using functions with an approximate gradient and deterministic algorithms

* 23.1.0
  - Fix bug in IndicatorBox proximal_conjugate
  - Allow CCPi Regulariser functions for non CIL object
  - Add norm for CompositionOperator
  - Refactor SIRT algorithm to make it more computationally and memory efficient
  - Optimisation in L2NormSquared
  - Added support for partitioner, when partitions have size 1
  - Fix for show_geometry bug for 2D data
  - FBP split processing bug fix - now respects panel origin set in geometry
  - Binner/Padder/Slicer bug fix - now respects panel origin set in geometry
  - Added warmstart capability to proximal evaluation of the CIL TotalVariation function.
  - Bug fix in the LinearOperator norm with an additional flag for the algorithm linearOperator.PowerMethod
  - Tidied up documentation in the framework folder

* 23.0.1
  - Fix bug with NikonReader requiring ROI to be set in constructor.

* 23.0.0
  - Partitioner is now able to create batches even if angle is not the outer dimension
  - Renamed `max_iteration_stop_cryterion` method in the Algorithm class to `max_iteration_stop_criterion`
  - Removed (previously deprecated) `very_verbose` parameter in `Algorithm`'s run method.
  - Removed (previously deprecated) `axpby` method in DataContainer.
  - Deprecate use of integer compression in NEXUSDataWriter.
  - Improved and tidied up documentation for all readers and writers, including hiding special members.
  - Use arguments instead of kwargs in all readers and writers with multiple kwargs, making documentation easier.
  - Update Apache2 License Headers.

* 22.2.0
  - BlockGeometry is iterable
  - Added `partition` to `AcquisitionData` to partition the data with 3 methods: `sequential`, `staggered` and `random_permutation`
  - TIGRE and ASTRA `ProjectionOperator` now support `BlockGeometry` as `acquisition_geometry` parameter, returning a `BlockOperator`
  - Added pre-set filters for `recon.FBP` and `recon.FDK`. Filters now include ram-lak, hamming, hann, cosine, shepp-logan.
  - Added RAWFileWriter to export data containers to raw files
  - Extended IndicatorBox to behave as IndicatorBoxPixelwise by passing masks in lower and upper bounds
  - Implemented IndicatorBox in numba and numpy
  - Dropped support for Python 3.6 and NumPy 1.15
  - Jenkins PR tests on Python 3.8 and NumPy 1.20
  - added yml file to create test environment
  - LeastSquares fixed docstring and unified gradient code when out is passed or not.
  - Add compression to 8bit and 16bit to TIFFWriter
  - Added convenience centre of rotation methods to `AcquisitionGeometry` class.
    - `get_centre_of_rotation()` calculates the centre of rotation of the system
    - `set_centre_of_rotation()` sets the system centre of rotation with an offset and angle
    - `set_centre_of_rotation_by_slice()` sets the system centre of rotation with offsets from two slices
  - Binner processor reworked:
    - Significant speed increase available via the C++ backend
    - Returned geometry is correctly offset where binning/cropping moves the origin
  - Slicer refactoring
    - Returned geometry is correctly offset where slicing/cropping moves the origin
  - Padder refactoring
    - Returned geometry is correctly offset where padding moves the origin
  - Github Actions:
    - update test python and numpy versions to 3.9 and 1.22
    - Update conda build action to v1.4.4
    - Fixes actions to run on ubuntu-20.04
    - Update version of upload_artifact github action to version 3.1.1
    - Update version of download_artifact github action to version 3.0.1
    - Update version of checkout github action to version 3.1.0
    - Update build-sphinx action to version 0.1.3
  - `io.utilities.HDF5_utilities` Added utility functions to browse hdf5 files and read datasets into numpy array
  - Implemented the analytical norm for GradientOperator
  - Added `ImageData.apply_circular_mask` method to mask out detector edge artefacts on reconstructed volumes
  - ROI selection, aspect ratio toggle and Play widget added to islicer
  - Add show1D display utility

* 22.1.0
  - use assert_allclose in test_DataContainer
  - added multiple colormaps to show2D
  - Fix segfault in GradientOperator due to parameter overflows on windows systems
  - Fix angle display precision and matplotlib warning for sinograms with show2D

* 22.0.0
  - Strongly convex functionality in TotalVariation and FGP_TV Functions
  - Refactored KullbackLeibler function class. Fix bug on gradient method for SIRF objects
  - Numba added as a CIL requirement
  - Simplify initialisation of `CentreOfRotation.ImageSharpness` with new parameter `backend`
  - Added ISTA algorithm. Improve inheritance of proximal gradient algorithms
  - Updated interface to `plugins.tigre`/`plugins.astra` `FBP` and `ProjectionOperator` classes
  - Update NikonDataReader to parse and set up geometry with: `ObjectTilt` `CentreOfRotationTop` and `CentreOfRotationBottom`
  - Cleaned up unit test structure and output
  - Removal of deprecated code:
    - AcquisitionGeometry `__init__` no longer returns a configured geometry, use factory `create` methods instead
    - `subset` method removed, use `get_slice` or `reorder` methods
    - NikonDataReader `normalize` kwarg removed, use `normalise`
    - Algorithms initialisation `x_init` kwarg removed, use `initial`
    - Removed deprecated numpy calls
  - DataProcessors use weak-reference to input data
  - Merged CIL-ASTRA code in to CIL repository simplifying test, build and install procedures
    - Modules not moved should be considered deprecated
    - CIL remains licensed as APACHE-2.0
    - Minor bug fixes to the CPU 2D Parallel-beam FBP
  - Add ndim property for DataContainer class
  - Fixes show_geometry compatibility issue with matplotlib 3.5
  - Added ZEISSDataReader with cone/parallel beam, slicing, TXM Functionality
  - Raise exception if filename or data haven't been set in NexusDataWriter
  - Fixes error when update_objective_interval is set to 0 in an algorithm run
  - Deprecated:
    - TXRMDataReader is deprecated in favour of ZEISSDataReader
  - GitHub Actions:
    - Update to version 0.1.1 of lauramurgatroyd/build-sphinx-action for building the documentation - ensures docs are always built from cil master

* 21.4.1
  - Removed prints from unittests and cleanup of unittest code.
  - CMake:
    - install script re-allows selection of non default install directory ([#1246](https://github.com/TomographicImaging/CIL/issues/1246))
  - TIFF writer uses logging
  - Added unittests for TIFF functionality

* 21.4.0
  - PEP 440 compliant version
  - CMake fix due to use of pip install.
  - Recon.FBP allows 'astra' backend
  - Fixed PowerMethod for square/non-square, complex/float matrices with stopping criterion.
  - CofR image_sharpness improved for large datasets
  - Geometry alignment fix for 2D datasets
  - CGLS update for sapyb to enable complex data, bugfix in use of initial
  - added sapyb and deprecated axpby. All algorithm updated to use sapyb.
  - Allow use of square brackets in file paths to TIFF and Nikon datasets

* 21.3.1
  - Added matplotlib version dependency to conda recipe
  - Fixed TIGRE wrappers for geometry with a virtual detector
  - Fixed TIGRE wrappers for cone-beam geometry with tilted rotation axis

* 21.3.0
  - Accelerated PDHG which handles strong convexity of functions
  - TotalVariation Function handles SIRF ImageData
  - Simulated datasets and volume added to DataExamples
  - TIGRE wrappers for parallel-beam geometry added
  - NEXUSWriter and NEXUSReader offer (8bit and 16bit) compression of data
  - show2D show_geom now return an object that can be saved with a `save` method
  - GradientOperator can be now used with SIRF DataContainers, both PET and MR
  - Add anisotropy in TotalVariation function
  - CCPi Regularisation plugin is refactored, only FGP_TV, FGP_dTV, TGV and TNV are exposed. Docstrings and functionality unit tests are added. Tests of the functions are meant to be in the CCPi-Regularisation toolkit itself.
  - Add dtype for ImageGeometry, AcquisitionGeometry, VectorGeometry, BlockGeometry
  - Fix GradientOperator to handle pseudo 2D CIL geometries
  - Created recon module with FBP and FDK using fast filtering library and TIGRE backprojectors
  - Added Intel IPP based library for filtering step of FBP
  - PDHG memory optimisation
  - ScaledFunction memory Optimisation
  - The github actions are merged into one action with multiple jobs
  - The conda build job uploads an artifact of the build tar.bz file which is later used by the documentation build job - which installs the package into a miniconda environment.
  - Documentation pages for recon, astra and cil-plugins are published.

* 21.2.0
  - add version string from git describe
  - add CCPi-Regularisation toolkit in unittests
  - show_geometry implemented to display AcquisitionGeometry objects, can be imported from utilities.display
  - CentreOfRotationCorrector.image_sharpness implemented which finds the rotation axis offset by maximising sharpness of a single slice reconstruction
  - Renamed CentreOfRotationCorrector.xcorr to CentreOfRotationCorrector.xcorrelation
  - Implemented Padder processor

* 21.1.0
  - Added TomoPhantom plugin to create 2D/3D + channel ImageData phantoms based on the TomoPhantom model library
  - Fixed bug in Zeiss reader geometry direction of rotation

* 21.0.0
  - Show2D now takes 4D datasets and slice information as input
  - TIGRE reconstruction package wrapped for cone-beam tomography
  - Datacontainers have get_slice method which returns a dataset with a single slice of the data
  - Datacontainers have reorder method which reorders the data in memory as requested, or for use with 'astra' or 'tigre'
  - Subset method has been deprecated
  - AcquisitionData and ImageData enforce requirement for a geometry on creation
  - New processors AbsorptionTransmissionConverter and TransmissionAbsorptionConverter to convert between Absorption and Transmission
  - Implemented Binner and Slicer processors
  - Implemented MaskGenerator and Masker processors

* 20.11.2
  - fixed windows build
  - NikonDataReader converts Nikon geometry to CIL geometry from xtekct file including detector and centre-or-rotation offsets
  - NexusdataReader supports files written with old versions of NexusDataWriter

* 20.11
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
