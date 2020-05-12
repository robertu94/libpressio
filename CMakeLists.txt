cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libpressio VERSION "0.39.1" LANGUAGES CXX C)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

set(default_cxx_standard 17)
if(NOT LIBPRESSIO_CXX_VERSION)
  set(LIBPRESSIO_CXX_VERSION ${default_cxx_standard} CACHE 
    STRING "Which version of the cxx standard to expect" FORCE)
  set_property(CACHE LIBPRESSIO_CXX_VERSION PROPERTY STRINGS 11 14 17)
endif()

include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "build libpressio as a shared library" ON)


include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)

check_cxx_compiler_flag("-fno-omit-frame-pointer" HAVE_NO_OMIT_FRAME_POINTER)
set(NO_OMIT_FRAME_POINTER_FLAG "")
if(HAVE_NO_OMIT_FRAME_POINTER)
  set(NO_OMIT_FRAME_POINTER_FLAG "-fno-omit-frame-pointer")
endif()


#check for c++17 features
function(check_cpp_standard api)
  string(TOUPPER "${api}" HEADER_SUFFIX)
  try_compile(
    COMPILED
    ${CMAKE_BINARY_DIR}
    SOURCES "${CMAKE_SOURCE_DIR}/checks/${api}.cc" 
    CXX_STANDARD ${LIBPRESSIO_CXX_VERSION}
    CXX_STANDARD_REQUIRED ON
  )
  message(STATUS "Checking for ${api}: ${COMPILED}")
  set(LIBPRESSIO_COMPAT_HAS_${HEADER_SUFFIX} ${COMPILED} PARENT_SCOPE)
endfunction()

check_cpp_standard(transform_reduce)
check_cpp_standard(exclusive_scan)
check_cpp_standard(exchange)
check_cpp_standard(rbeginend)
check_cpp_standard(optional)
check_cpp_standard(variant)
check_cpp_standard(make_unique)
check_cpp_standard(conjunction)
check_cpp_standard(multiplies)


set(LIBPRESSIO_FEATURES "")
set(LIBPRESSIO_COMPRESSORS "noop")

add_library(libpressio
  #core implementation
  ./src/pressio.cc
  ./src/pressio_configurable.cc
  ./src/pressio_versionable.cc
  ./src/pressio_errorable.cc
  ./src/pressio_compressor.cc
  ./src/pressio_data.cc
  ./src/pressio_dtype.cc
  ./src/pressio_metrics.cc
  ./src/pressio_option.cc
  ./src/pressio_options.cc
  ./src/pressio_options_iter.cc

  #plugins
  ./src/plugins/compressors/compressor_base.cc
  ./src/plugins/compressors/noop.cc
  ./src/plugins/compressors/sampling.cc
  ./src/plugins/compressors/resize.cc
  ./src/plugins/compressors/transpose.cc
  ./src/plugins/metrics/composite.cc
  ./src/plugins/metrics/external.cc
  ./src/plugins/metrics/external_forkexec.cc
  ./src/plugins/metrics/metrics_base.cc
  ./src/plugins/metrics/size.cc
  ./src/plugins/metrics/time.cc
  ./src/plugins/metrics/error_stat.cc
  ./src/plugins/metrics/pearsons.cc
  ./src/plugins/metrics/spatial_error.cc
  ./src/plugins/io/posix.cc
  ./src/plugins/io/noop.cc
  ./src/plugins/io/csv.cc
  ./src/plugins/io/io.cc
  ./src/plugins/io/select.cc

  #public headers
  include/libpressio.h
  include/libpressio_ext/cpp/configurable.h
  include/libpressio_ext/cpp/versionable.h
  include/libpressio_ext/cpp/errorable.h
  include/libpressio_ext/cpp/compressor.h
  include/libpressio_ext/cpp/data.h
  include/libpressio_ext/cpp/libpressio.h
  include/libpressio_ext/cpp/metrics.h
  include/libpressio_ext/cpp/options.h
  include/libpressio_ext/cpp/pressio.h
  include/libpressio_ext/cpp/printers.h
  include/libpressio_ext/cpp/io.h
  include/libpressio_ext/io/posix.h
  include/libpressio_ext/io/pressio_io.h
  include/pressio.h
  include/pressio_compressor.h
  include/pressio_data.h
  include/pressio_dtype.h
  include/pressio_metrics.h
  include/pressio_option.h
  include/pressio_options.h
  include/pressio_options_iter.h
  )

target_include_directories(libpressio
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libpressio>
  )
target_compile_options(libpressio PRIVATE 
  $<$<CONFIG:Debug>: -Wall -Werror -Wextra -Wpedantic>
  $<$<CONFIG:RelWithDebInfo>: ${NO_OMIT_FRAME_POINTER_FLAG}>
  )
target_compile_features(libpressio PUBLIC cxx_std_${LIBPRESSIO_CXX_VERSION})

if(NOT (LIBPRESSIO_COMPAT_HAS_VARIANT OR LIBPRESSIO_COMPAT_HAS_OPTIONAL))
  find_package(Boost COMPONENTS REQUIRED)
  target_link_libraries(libpressio PUBLIC Boost::boost)
endif()

find_package(PkgConfig REQUIRED)

option(LIBPRESSIO_HAS_OPENMP "accerate some plugins with OpenMP" OFF)
if(LIBPRESSIO_HAS_OPENMP)
  find_package(OpenMP)
  target_link_libraries(libpressio PRIVATE OpenMP::OpenMP_CXX)
endif()

option(LIBPRESSIO_HAS_MGARD "build the MGARD plugin" OFF)
if(LIBPRESSIO_HAS_MGARD)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} mgard")
  find_package(mgard REQUIRED)
  check_include_file_cxx(mgard_api_float.h LIBPRESSIO_MGARD_NEED_FLOAT_HEADER
    -I $<TARGET_PROPERTY:mgard::mgard,INTERFACE_INCLUDE_DIRECTORIES>)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/mgard_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE mgard::mgard)
endif()

option(LIBPRESSIO_HAS_ZFP "build the ZFP plugin" OFF)
if(LIBPRESSIO_HAS_ZFP)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} zfp")
  find_package(ZFP REQUIRED)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/zfp_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE zfp::zfp)
endif()

option(LIBPRESSIO_HAS_SZ "build the SZ plugin" OFF)
if(LIBPRESSIO_HAS_SZ)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} sz")
  find_package(SZ REQUIRED)
  find_package(ZLIB REQUIRED)
  pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
  pkg_search_module(FFTW3 IMPORTED_TARGET GLOBAL fftw3)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE SZ)
endif()

option(LIBPRESSIO_HAS_BLOSC "build the BLOSC plugin" OFF)
if(LIBPRESSIO_HAS_BLOSC)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} blosc")
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/blosc_plugin.cc
    )
  pkg_search_module(BLOSC IMPORTED_TARGET GLOBAL blosc)
  if(BLOSC_FOUND)
    target_link_libraries(libpressio PRIVATE PkgConfig::BLOSC)
  else()
          find_library(BLOSC_LIBRARIES blosc)
          find_file(BLOSC_INCLUDE_FILE blosc.h)
          if(NOT BLOSC_LIBRARIES)
            message(FATAL_ERROR "Failed to find blosc and it was enabled")
          endif()
          get_filename_component(BLOSC_INCLUDE_DIRS ${BLOSC_INCLUDE_FILE} DIRECTORY)
          target_link_libraries(libpressio PRIVATE ${BLOSC_LIBRARIES})
          target_include_directories(libpressio PRIVATE ${BLOSC_INCLUDE_DIRS})
          
  endif()

endif()


option(LIBPRESSIO_HAS_MAGICK "build the image magick plugin" OFF)
if(LIBPRESSIO_HAS_MAGICK)
  pkg_search_module(Magick++ IMPORTED_TARGET GLOBAL Magick++)
  try_compile(
    LIBPRESSIO_COMPAT_HAS_IMAGEMAGICK_LONGLONG
    ${CMAKE_BINARY_DIR}
    SOURCES "${CMAKE_SOURCE_DIR}/checks/imagemagick.cc"
    LINK_LIBRARIES PkgConfig::Magick++
  )
  message(STATUS "Checking for Magick::LongLong: ${LIBPRESSIO_COMPAT_HAS_IMAGEMAGICK_LONGLONG}")
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} magick")
  target_link_libraries(libpressio PRIVATE PkgConfig::Magick++)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/magick++.cc
    )
endif()

option(LIBPRESSIO_HAS_HDF "build the hdf5 io plugin" OFF)
if(LIBPRESSIO_HAS_HDF)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} hdf5")
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_link_libraries(libpressio PRIVATE ${HDF5_C_LIBRARIES})
  target_include_directories(libpressio PRIVATE ${HDF5_C_INCLUDE_DIRS})
  target_compile_definitions(libpressio PRIVATE ${HDF5_C_DEFINITIONS})
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/hdf5.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/io/hdf5.h
    )
endif()

option(LIBPRESSIO_HAS_FPZIP "build the fpzip plugin" OFF)
if(LIBPRESSIO_HAS_FPZIP)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} fpzip")
  find_package(fpzip)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/fpzip_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE fpzip::fpzip)

endif()

option(LIBPRESSIO_HAS_MPI "build MPI spawn external method" OFF)
if(LIBPRESSIO_HAS_MPI)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} mpi")
  find_package(MPI COMPONENTS CXX)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/metrics/external_mpispawn.cc
    )
  target_link_libraries(libpressio PRIVATE MPI::MPI_CXX)
endif()

option(LIBPRESSIO_HAS_PETSC "build the petsc io plugin" OFF)
if(LIBPRESSIO_HAS_PETSC)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} petsc")
  pkg_search_module(PETSc IMPORTED_TARGET PETSc)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/petsc.cc
    )
  target_link_libraries(libpressio PRIVATE PkgConfig::PETSc)
endif()

option(LIBPRESSIO_HAS_LUA "build multi-objective code" OFF)
if(LIBPRESSIO_HAS_LUA)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} lua")
  find_package(sol2 REQUIRED)
  pkg_search_module(Lua IMPORTED_TARGET lua luajit)
  if(Lua_FOUND)
    message(INFO "lua from pkgconfig")
    target_link_libraries(libpressio PRIVATE PkgConfig::Lua)
  else()
    message(INFO "lua from cmake")
    find_package(Lua)
    target_link_libraries(libpressio PRIVATE ${LUA_LIBRARIES})
    target_include_directories(libpressio PRIVATE ${LUA_INCLUDE_DIR})
  endif()
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/pressio_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/pressio_version.h
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio.pc
  @ONLY
  )

option(USE_CLANG_TIDY "include clang-tidy warnings in the build log" OFF)
if(USE_CLANG_TIDY)
  find_program(CLANG_TIDY clang-tidy)
  set_target_properties(libpressio PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY}")
  set_target_properties(libpressio PROPERTIES C_CLANG_TIDY "${CLANG_TIDY}")
endif()

option(USE_INCLUDE_WHAT_YOU_USE "include include-what-you-use warnings in the build log" OFF)
if(USE_INCLUDE_WHAT_YOU_USE)
  find_program(INCLUDE_WHAT_YOU_USE include-what-you-use)
  set_target_properties(libpressio PROPERTIES CXX_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
  set_target_properties(libpressio PROPERTIES C_INCLUDE_WHAT_YOU_USE "${INCLUDE_WHAT_YOU_USE}")
endif()


export(TARGETS libpressio NAMESPACE LibPressio:: FILE LibPressio.cmake)
install(TARGETS libpressio EXPORT LibPressioConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT LibPressioConfig NAMESPACE LibPressio:: DESTINATION share/LibPressio/cmake)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/pressio_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

option(BUILD_PYTHON_WRAPPER "build python wrapper" OFF)
if(BUILD_PYTHON_WRAPPER)
  add_subdirectory(swig)
endif()

option(BUILD_DOCS "build the documetation" OFF)
if(BUILD_DOCS)
  find_package(Doxygen REQUIRED dot)
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_MAN_LINKS YES)
  set(DOXYGEN_GENERATE_MAN YES)
  set(DOXYGEN_EXTRACT_LOCAL_METHODS YES)
  set(DOXYGEN_EXTRACT_STATIC YES)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  doxygen_add_docs(
    docs
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/docs
    ${PROJECT_SOURCE_DIR}/include
    COMMENT "Generate Documenation"
  )
endif()

option(BUILD_DOCKER_CONTAINERS "build docker containers for testing" OFF)
if(BUILD_DOCKER_CONTAINERS)
  
  foreach(CONTAINER Fedora Ubuntu18.04 CentOS)
    set(BuildSentinel ${CMAKE_BINARY_DIR}/${CONTAINER}-built)
    set(Dockerfile docker/Dockerfile-${CONTAINER})
    string(TOLOWER "libpressio-${CONTAINER}" CONTAINER_TAG)
    add_custom_command(OUTPUT ${BuildSentinel}
      COMMAND docker build -t ${CONTAINER_TAG} -f ${Dockerfile} .
      COMMAND touch ${BuildSentinel}
      MAIN_DEPENDENCY ${Dockerfile}
      DEPENDS libpressio
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
      COMMENT "DOCKER ${Dockerfile}"
      )
    list(APPEND DOCKER_CONTAINERS ${BuildSentinel})
  endforeach()
  add_custom_target(docker DEPENDS ${DOCKER_CONTAINERS} COMMENT "building docker containers")
endif()
