* 2x.x.x
  - added sapyb and deprecated axpby

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
  - Show2D now takes 4D datasets and slice infomation as input
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
