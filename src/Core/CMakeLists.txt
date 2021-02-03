
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_STANDARD_REQUIRED ON)
set (CMAKE_BUILD_TYPE Release)

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



add_library(cilacc SHARED ${CMAKE_CURRENT_SOURCE_DIR}/utilities.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/FiniteDifferenceLibrary.cpp
                          ${CMAKE_CURRENT_SOURCE_DIR}/axpby.cpp )

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL "3.9.0") 
  set (OpenMP_EXE_LINKER_FLAGS OpenMP::OpenMP_CXX)
else()
  message(WARNING "Your CMake version is old. OpenMP linking flags  might be incorrect.")
  # need to explicitly set this. Definitely for gcc, hopefully also for other systems.
  # See https://gitlab.kitware.com/cmake/cmake/issues/15392
  set (OpenMP_EXE_LINKER_FLAGS ${OpenMP_C_FLAGS})
endif()

target_link_libraries(cilacc ${OpenMP_EXE_LINKER_FLAGS} )
include_directories(cilacc PUBLIC 
				  ${CMAKE_CURRENT_SOURCE_DIR}/include
		          )

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


