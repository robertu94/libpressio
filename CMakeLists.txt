cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(liblossy VERSION "0.3.0" LANGUAGES CXX C)

enable_testing()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(GNUInstallDirs)
find_package(ZFP)
find_package(PkgConfig)
pkg_search_module(SZ IMPORTED_TARGET GLOBAL sz)


set(LIBLOSSY_FEATURES "sz zfp")
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/include/lossy_version.h.in
  ${CMAKE_CURRENT_BINARY_DIR}/include/lossy_version.h
  )
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/liblossy.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/liblossy.pc
  @ONLY
  )

add_library(liblossy
  #public headers
  ./include/lossy_data.h
  ./include/lossy_compressor.h
  ./include/lossy.h
  ./include/lossy_options.h
  ./include/lossy_dtype.h
  ./include/lossy_option.h

  #implementations
  ./src/lossy_options.cc
  ./src/lossy_option.cc
  ./src/plugins/liblossy_plugin.cc
  ./src/plugins/sz_plugin.cc
  ./src/plugins/zfp_plugin.cc
  ./src/lossy_data.cc
  ./src/lossy_dtype.cc
  ./src/lossy.cc
  ./src/lossy_compressor.cc
  ./src/lossy_options_iter.cc


  #private headers
  ./src/plugins.h
  ./src/plugins/liblossy_plugin.h
  ./src/lossy_compressor_impl.h
  ./src/lossy_options_impl.h
  )

target_include_directories(liblossy
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/include
  )
target_compile_options(liblossy PRIVATE $<$<CONFIG:Debug>: -Wall -Werror -Wextra -Wpedantic>)
target_link_libraries(liblossy PUBLIC zfp::zfp PkgConfig::SZ)
set_target_properties(liblossy PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON) 

install(TARGETS liblossy EXPORT LibLossyConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT LibLossyConfig DESTINATION share/LibLossy/cmake)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/liblossy)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/lossy_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/liblossy)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/liblossy.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
export(TARGETS liblossy FILE LibLossy.cmake)

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
    ${PROJECT_SOURCE_DIR}/src/plugins/liblossy_plugin.h
    COMMENT "Generate Documenation"
  )
endif()
