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
find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

find_package(nanobind CONFIG REQUIRED)

find_package(IPP REQUIRED CONFIG)
find_package(OpenMP REQUIRED)



if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNUCC")
  # appends some flags
  add_compile_options(-ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec)
  # add_compile_options(-march=native -mavx )
endif()

message ("OpenMP_CXX_FLAGS ${OpenMP_CXX_FLAGS}")
#add_definitions(${OpenMP_CXX_FLAGS})

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

list(APPEND cilacc_SOURCES utilities.cpp axpby.cpp FiniteDifferenceLibrary.cpp nanobind.cpp)
list(APPEND cilacc_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/include)
list(APPEND cilacc_LIBRARIES ${OpenMP_EXE_LINKER_FLAGS})

if(IPP_FOUND)
  list(APPEND cilacc_SOURCES FBP_filtering.cpp Binning.cpp)
  list(APPEND cilacc_INCLUDES ${IPP_INCLUDE_DIRS})
  list(APPEND cilacc_LIBRARIES ${IPP_LIBRARIES})
  message(STATUS "IPP Libraries Found")
else()
  message(WARNING "IPP libraries not found, FBP_filtering not in build")
endif()
message(STATUS ${cilacc_INCLUDES})
message(STATUS ${IPP_LIBRARIES})

nanobind_add_module(
	cilacc
	NB_STATIC
	${cilacc_SOURCES}
)
include_directories(cilacc PUBLIC ${cilacc_INCLUDES})
target_link_libraries(cilacc PUBLIC ${cilacc_LIBRARIES})

install(TARGETS cilacc LIBRARY DESTINATION cil)
