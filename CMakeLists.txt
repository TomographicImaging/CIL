cmake_minimum_required(VERSION 3.4)

project (cil LANGUAGES C)

set(CIL_VERSION $ENV{CIL_VERSION} CACHE STRING "CIL VERSION" FORCE)
add_subdirectory(src/Core)
add_subdirectory(Wrappers/Python)
