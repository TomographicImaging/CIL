cmake_minimum_required(VERSION 3.4)

project (cil LANGUAGES CXX)

# If APPLE don't use @rpath
if (APPLE)
  option(SHARED_LIBS_ABS_PATH "Force shared libraries to be installed with absolute paths (as opposed to rpaths)" ON)
  mark_as_advanced( SHARED_LIBS_ABS_PATH )  
  if (SHARED_LIBS_ABS_PATH)
    # Set install_name_dir as the absolute path to install_prefix/lib
    GET_FILENAME_COMPONENT(CMAKE_INSTALL_NAME_DIR ${CMAKE_INSTALL_PREFIX}/lib REALPATH)
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
  endif(SHARED_LIBS_ABS_PATH)
endif(APPLE)

set(CMAKE_BUILD_TYPE "Release")

set(CIL_VERSION $ENV{CIL_VERSION})
add_subdirectory(src/Core)
add_subdirectory(Wrappers/Python)
