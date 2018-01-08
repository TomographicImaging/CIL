#!/usr/bin/env python

import setuptools
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import os
import numpy
import platform	
import sys

cil_version=os.environ['CIL_VERSION']
if  cil_version == '':
    print("Please set the environmental variable CIL_VERSION")
    sys.exit(1)

library_include_path = []
library_lib_path = []
try:
    library_include_path = [ os.environ['LIBRARY_INC'] ]
    library_lib_path = [ os.environ['LIBRARY_LIB'] ]
except:
    if platform.system() == 'Windows':
        pass
    else:
        try:
           library_include_path = [ os.environ['PREFIX']+'/include' ]
           library_lib_path = [ os.environ['PREFiX']+'/lib' ]
        except:
           pass
    pass
extra_include_dirs = [numpy.get_include()]
extra_library_dirs = []
extra_compile_args = []
extra_link_args = []
extra_libraries = []

if platform.system() == 'Windows':
   extra_compile_args += ['/DWIN32','/EHsc','/DBOOST_ALL_NO_LIB', 
   '/openmp','/DHAS_TIFF','/DCCPiReconstructionIterative_EXPORTS']   
   extra_include_dirs += ["..\\..\\Core\\src\\","..\\..\\Core\\src\\Algorithms","..\\..\\Core\\src\\Readers", "."]
   extra_include_dirs += library_include_path
   extra_library_dirs += library_lib_path
   extra_libraries    += ['tiff' , 'cilrec']
   if sys.version_info.major == 3 :   
       extra_libraries += ['boost_python3-vc140-mt-1_64', 'boost_numpy3-vc140-mt-1_64']
   else:
       extra_libraries += ['boost_python-vc90-mt-1_64', 'boost_numpy-vc90-mt-1_64']   
else:
   extra_include_dirs += ["../../Core/src/","../../Core/src/Algorithms","../../Core/src/Readers", "."]
   extra_include_dirs += library_include_path
   extra_compile_args += ['-fopenmp','-O2', '-funsigned-char', '-Wall','-Wl,--no-undefined','-DHAS_TIFF','-DCCPiReconstructionIterative_EXPORTS']  
   extra_libraries    += ['tiff' , 'cilrec'] 
   if sys.version_info.major == 3 :
       extra_libraries += ['boost_python3', 'boost_numpy3','gomp']
   else:
       extra_libraries += ['boost_python', 'boost_numpy','gomp']


setup(
  name='ccpi-reconstruction',
	description='This is a CCPi Core Imaging Library package for Iterative Reconstruction codes',
	version=cil_version,
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("ccpi.reconstruction.parallelbeam",
                             sources=[  "src/diamond_module.cpp",
                                        "src/diamond_wrapper.cpp"],
                             include_dirs=extra_include_dirs, library_dirs=extra_library_dirs, extra_compile_args=extra_compile_args, libraries=extra_libraries, extra_link_args=extra_link_args ),
                             Extension("ccpi.reconstruction.conebeam",
                             sources=[  "src/conebeam_module.cpp",
                                        "src/conebeam_wrapper.cpp"],
                             include_dirs=extra_include_dirs, library_dirs=extra_library_dirs, extra_compile_args=extra_compile_args, libraries=extra_libraries )                             ],
	zip_safe = False,
	packages = {'ccpi','ccpi.reconstruction'}
)
