#  Copyright 2019 United Kingdom Research and Innovation
#  Copyright 2019 The University of Manchester
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
# Authors:
# CIL Developers, listed at: https://github.com/TomographicImaging/CIL/blob/master/NOTICE.txt
# Richard Brown (UCL)

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

message ("OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
find_package(OpenMP REQUIRED)
add_definitions(${OpenMP_CXX_FLAGS})


if (WIN32)
	set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Ddll_EXPORTS")
endif()

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNUCC")
  # appends some flags
  add_compile_options(-ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec)
  # add_compile_options(-march=native -mavx )
endif()

message("CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS}")
message("CMAKE_EXE_LINKER_FLAGS ${CMAKE_EXE_LINKER_FLAGS}")
message("CMAKE_SHARED_LINKER_FLAGS ${CMAKE_SHARED_LINKER_FLAGS}")
message("CMAKE_STATIC_LINKER_FLAGS ${CMAKE_STATIC_LINKER_FLAGS}")

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0")
  set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
else()
  message(WARNING "Your CMake version is old. OpenMP linking flags  might be incorrect.")
  # need to explicitly set this. Definitely for gcc, hopefully also for other systems.
  # See https://gitlab.kitware.com/cmake/cmake/issues/15392
  set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS})
endif()

if (UNIX)
  find_library(IPP_CORE libippcore.so)
  find_library(IPP_S libipps.so)
  find_library(IPP_VM libippvm.so)
  find_library(IPP_I libippi.so)
elseif(WIN32)
  find_library(IPP_CORE ippcore)
  find_library(IPP_S ipps)
  find_library(IPP_VM ippvm)
  find_library(IPP_I ippi)
endif()

if(IPP_CORE AND IPP_S AND IPP_VM AND IPP_I)
  set(USE_IPP TRUE)
  message("IPP libraries found")
  file(REAL_PATH ${IPP_CORE}/../.. LIB_BASE)
  find_file(IPP_INCLUDE ipp.h ${LIB_BASE} PATH_SUFFIXES include include/ipp)
  cmake_path(GET IPP_INCLUDE PARENT_PATH IPP_INCLUDE)
else()
  set(USE_IPP FALSE)
  message("IPP libraries not found, FBP_filtering not in build")
endif()

if(USE_IPP)
  add_library(cilacc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/utilities.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/axpby.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/FiniteDifferenceLibrary.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/FBP_filtering.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/Binning.cpp)


  target_link_libraries(cilacc ${OpenMP_EXE_LINKER_FLAGS} ${IPP_CORE} ${IPP_S} ${IPP_I} ${IPP_VM})
  include_directories(cilacc PUBLIC ${IPP_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include)
else()
  add_library(cilacc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/utilities.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/axpby.cpp
                            ${CMAKE_CURRENT_SOURCE_DIR}/FiniteDifferenceLibrary.cpp )

  target_link_libraries(cilacc ${OpenMP_EXE_LINKER_FLAGS})
  include_directories(cilacc PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

## Install
#include(GNUInstallDirs)
#install(TARGETS cilacc
#	RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib
#	CONFIGURATIONS ${CMAKE_BUILD_TYPE}
#	)

if (UNIX)
message ("I'd install into ${CMAKE_INSTALL_PREFIX}/lib")
install(TARGETS cilacc
	LIBRARY DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE}
	)
elseif(WIN32)
message ("I'd install into ${CMAKE_INSTALL_PREFIX} lib bin")
  install(TARGETS cilacc
	RUNTIME DESTINATION Library/bin
	ARCHIVE DESTINATION lib
	CONFIGURATIONS ${CMAKE_BUILD_TYPE}
	)
endif()

install(DIRECTORY ${PROJECT_SOURCE_DIR}/src/Core/include/
        DESTINATION ${CMAKE_INSTALL_PREFIX}/include/cil)
