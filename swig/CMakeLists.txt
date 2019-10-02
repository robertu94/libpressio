cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
find_package(Python COMPONENTS Interpreter Development NumPy)
find_package(SWIG)
if(SWIG_FOUND AND PYTHON_FOUND)
  include(UseSWIG)


  function(build_pressio_swig_module)
    get_filename_component(libname ${ARGV0} NAME_WE)
    set_property(SOURCE ${ARGV0} PROPERTY CPLUSPLUS ON)
    set_property(SOURCE ${ARGV0} PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)
    swig_add_library(
      ${libname}
      LANGUAGE python
      SOURCES ${ARGV}
      )
    set_property(TARGET ${libname} PROPERTY SWIG_COMPILE_DEFINITIONS "SWIGWORDSIZE64")
    target_link_libraries(${libname} PUBLIC libpressio Python::Python Python::NumPy)
    target_include_directories(${libname} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    get_property(swig_generated_module TARGET ${libname} PROPERTY SWIG_SUPPORT_FILES)

    install(TARGETS ${libname} DESTINATION ${Python_SITELIB})
    install(FILES ${swig_generated_module} DESTINATION ${Python_SITELIB})

  endfunction()


  build_pressio_swig_module(pypressio.i)
  build_pressio_swig_module(pypressio_sz.i)
  build_pressio_swig_module(pypressio_zfp.i)


endif()

