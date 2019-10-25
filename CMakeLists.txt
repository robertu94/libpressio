cmake_minimum_required(VERSION 3.13 FATAL_ERROR)
project(libpressio VERSION "0.18.2" LANGUAGES CXX C)

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

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "build libpressio as a shared library" ON)


include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

check_cxx_compiler_flag("-fno-omit-frame-pointer" HAVE_NO_OMIT_FRAME_POINTER)
set(NO_OMIT_FRAME_POINTER_FLAG "")
if(HAVE_NO_OMIT_FRAME_POINTER)
  set(NO_OMIT_FRAME_POINTER_FLAG "-fno-omit-frame-pointer")
endif()

#check for c++17 numeric functions
function(check_cpp_standard api)
  string(TOUPPER "${api}" HEADER_SUFFIX)
  try_compile(
    COMPILED
    ${CMAKE_BINARY_DIR}
    SOURCES "${CMAKE_SOURCE_DIR}/checks/${api}.cc" 
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
  )
  message(STATUS "Checking for ${api}: ${COMPILED}")
  set(LIBPRESSIO_COMPAT_HAS_${HEADER_SUFFIX} ${COMPILED} PARENT_SCOPE)
endfunction()

check_cpp_standard(transform_reduce)
check_cpp_standard(exclusive_scan)


set(LIBPRESSIO_FEATURES "")
set(LIBPRESSIO_COMPRESSORS "")

add_library(libpressio
  #core implementation
  ./src/pressio.cc
  ./src/pressio_compressor.cc
  ./src/pressio_data.cc
  ./src/pressio_dtype.cc
  ./src/pressio_metrics.cc
  ./src/pressio_option.cc
  ./src/pressio_options.cc
  ./src/pressio_options_iter.cc

  #plugins
  ./src/plugins/compressors/compressor_base.cc
  ./src/plugins/metrics/composite.cc
  ./src/plugins/metrics/metrics_base.cc
  ./src/plugins/metrics/size.cc
  ./src/plugins/metrics/time.cc
  ./src/plugins/metrics/error_stat.cc
  ./src/plugins/io/posix.cc

  #public headers
  include/libpressio.h
  include/libpressio_ext/cpp/compressor.h
  include/libpressio_ext/cpp/metrics.h
  include/libpressio_ext/cpp/plugins.h
  include/libpressio_ext/io/posix.h
  include/pressio.h
  include/pressio_compressor.h
  include/pressio_data.h
  include/pressio_dtype.h
  include/pressio_metrics.h
  include/pressio_option.h
  include/pressio_options.h
  include/pressio_options_iter.h

  #private headers
  src/pressio_compressor_impl.h
  )

target_include_directories(libpressio
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libpressio>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_compile_options(libpressio PRIVATE 
  $<$<CONFIG:Debug>: -Wall -Werror -Wextra -Wpedantic>
  $<$<CONFIG:RelWithDebInfo>: ${NO_OMIT_FRAME_POINTER_FLAG}>
  )
target_compile_features(libpressio PUBLIC cxx_std_17)

option(LIBPRESSIO_HAS_MGARD "build the MGARD plugin" OFF)
if(LIBPRESSIO_HAS_MGARD)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} mgard")
  find_package(mgard REQUIRED)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/mgard_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/compressors/mgard.h
    )
  target_link_libraries(libpressio PUBLIC mgard::mgard)
endif()

option(LIBPRESSIO_HAS_ZFP "build the ZFP plugin" ON)
if(LIBPRESSIO_HAS_ZFP)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} zfp")
  find_package(ZFP REQUIRED)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/zfp_plugin.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/compressors/zfp.h
    )
  target_link_libraries(libpressio PUBLIC zfp::zfp)
endif()

option(LIBPRESSIO_HAS_SZ "build the SZ plugin" ON)
if(LIBPRESSIO_HAS_SZ)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} sz")
  find_package(SZ REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
  pkg_search_module(FFTW3 IMPORTED_TARGET GLOBAL fftw3)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_plugin.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/compressors/sz.h
    )
  target_link_libraries(libpressio PUBLIC SZ)
endif()

option(LIBPRESSIO_HAS_HDF "build the hdf5 io plugin" ON)
if(LIBPRESSIO_HAS_HDF)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} hdf5")
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_link_libraries(libpressio PUBLIC ${HDF5_C_LIBRARIES})
  target_include_directories(libpressio PUBLIC ${HDF5_C_INCLUDE_DIRS})
  target_compile_definitions(libpressio PUBLIC ${HDF5_C_DEFINITIONS})
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/hdf5.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/io/hdf5.h
    )
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pressio_version.h.in
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

option(BUILD_TESTS "build the test cases and examples" OFF)
if(BUILD_TESTS)
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
    ${PROJECT_SOURCE_DIR}/include
    COMMENT "Generate Documenation"
  )
  
endif()
