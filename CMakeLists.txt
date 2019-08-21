cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(libpressio VERSION "0.8.0" LANGUAGES CXX C)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "build libpressio as a shared library" ON)

include(GNUInstallDirs)
find_package(ZFP REQUIRED)
find_package(SZ REQUIRED)
find_package(ZLIB REQUIRED)
find_package(PkgConfig REQUIRED)
find_package(Python COMPONENTS Interpreter Development NumPy)
pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)


set(LIBPRESSIO_FEATURES "sz zfp")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/pressio_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/pressio_version.h
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio.pc
  @ONLY
  )

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
  ./src/plugins/compressors/sz_plugin.cc
  ./src/plugins/compressors/zfp_plugin.cc
  ./src/plugins/metrics/composite.cc
  ./src/plugins/metrics/metrics_base.cc
  ./src/plugins/metrics/size.cc
  ./src/plugins/metrics/time.cc
  ./src/plugins/io/posix.cc

  #public headers
  include/libpressio.h
  include/libpressio_ext/compressors/sz.h
  include/libpressio_ext/compressors/zfp.h
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
  $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_compile_options(libpressio PRIVATE $<$<CONFIG:Debug>: -Wall -Werror -Wextra -Wpedantic>)
target_link_libraries(libpressio PUBLIC zfp::zfp SZ)
target_compile_features(libpressio PUBLIC cxx_std_17)

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
