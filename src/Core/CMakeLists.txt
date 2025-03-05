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
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT DEFINED CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

message ("OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
find_package(OpenMP REQUIRED)
add_definitions(${OpenMP_CXX_FLAGS})

cmake_minimum_required(VERSION 3.15)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module OPTIONAL_COMPONENTS Development.SABIModule)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(nanobind CONFIG REQUIRED)


####Needed to find IPP libraries 
execute_process(
	COMMAND "${Python_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_paths()[\"data\"])"
	OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE DATA_DIR)
list(APPEND CMAKE_PREFIX_PATH ${DATA_DIR}/cmake)
find_package(IPP REQUIRED CONFIG)



if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNUCC")
  # appends some flags
  add_compile_options(-ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec)
  # add_compile_options(-march=native -mavx )
endif()


if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0")
  set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
else()
  message(WARNING "Your CMake version is old. OpenMP linking flags  might be incorrect.")
  # need to explicitly set this. Definitely for gcc, hopefully also for other systems.
  # See https://gitlab.kitware.com/cmake/cmake/issues/15392
  set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS})
endif()




nanobind_add_module(
	cilacc
	NB_STATIC
	${CMAKE_CURRENT_SOURCE_DIR}/nanobind.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/include/axpby.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/Binning.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/FBP_filtering.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/FiniteDifferenceLibrary.h
	${CMAKE_CURRENT_SOURCE_DIR}/include/utilities.h
	${CMAKE_CURRENT_SOURCE_DIR}/axpby.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/Binning.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/FBP_filtering.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/FiniteDifferenceLibrary.cpp
	${CMAKE_CURRENT_SOURCE_DIR}/utilities.cpp
)
target_link_libraries(cilacc PUBLIC ${OpenMP_EXE_LINKER_FLAGS} ${IPP_LIBRARIES})
include_directories(cilacc PUBLIC 
	${CMAKE_CURRENT_SOURCE_DIR}/include/
)

install(TARGETS cilacc LIBRARY DESTINATION cil)
