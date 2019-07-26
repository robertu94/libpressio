cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(libpressio VERSION "0.4.0" LANGUAGES CXX C)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
find_package(ZFP)
find_package(PkgConfig)
pkg_search_module(SZ IMPORTED_TARGET GLOBAL sz)


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
  #public headers
  ./include/pressio_data.h
  ./include/pressio_compressor.h
  ./include/pressio.h
  ./include/pressio_options.h
  ./include/pressio_dtype.h
  ./include/pressio_option.h

  #implementations
  ./src/pressio_options.cc
  ./src/pressio_option.cc
  ./src/plugins/libpressio_plugin.cc
  ./src/plugins/sz_plugin.cc
  ./src/plugins/zfp_plugin.cc
  ./src/pressio_data.cc
  ./src/pressio_dtype.cc
  ./src/pressio.cc
  ./src/pressio_compressor.cc
  ./src/pressio_options_iter.cc


  #private headers
  ./src/plugins.h
  ./src/plugins/libpressio_plugin.h
  ./src/pressio_compressor_impl.h
  ./src/pressio_options_impl.h
  )

target_include_directories(libpressio
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_compile_options(libpressio PRIVATE $<$<CONFIG:Debug>: -Wall -Werror -Wextra -Wpedantic>)
target_link_libraries(libpressio PUBLIC zfp::zfp PkgConfig::SZ)
set_target_properties(libpressio PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON) 

install(TARGETS libpressio EXPORT LibPressioConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT LibPressioConfig DESTINATION share/LibPressio/cmake)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/pressio_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
export(TARGETS libpressio FILE LibPressio.cmake)

option(BUILD_TESTS "build the test cases and examples" OFF)

if(BUILD_TESTS)
  add_subdirectory(test)
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
    ${PROJECT_SOURCE_DIR}/src/plugins/libpressio_plugin.h
    COMMENT "Generate Documenation"
  )
endif()
