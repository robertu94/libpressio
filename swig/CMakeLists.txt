cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
find_package(Python COMPONENTS Interpreter Development NumPy)
find_package(SWIG)
if(SWIG_FOUND AND PYTHON_FOUND)
  include(UseSWIG)


  function(build_pressio_swig_module)
    get_filename_component(filelibname ${ARGV0} NAME_WE)
    set_property(SOURCE ${ARGV0} PROPERTY CPLUSPLUS ON)
    set_property(SOURCE ${ARGV0} PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)
    set_property(SOURCE ${ARGV0}.i PROPERTY SWIG_MODULE_NAME ${filelibname})
    swig_add_library(
      ${filelibname}
      LANGUAGE python
      SOURCES ${ARGV}
      )
    set_property(TARGET ${filelibname} PROPERTY SWIG_COMPILE_DEFINITIONS "SWIGWORDSIZE64")
    target_link_libraries(${filelibname} PUBLIC libpressio Python::Python Python::NumPy)
    target_include_directories(${filelibname} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    get_property(swig_generated_module TARGET ${filelibname} PROPERTY SWIG_SUPPORT_FILES)

    install(TARGETS ${filelibname} DESTINATION ${Python_SITELIB})
    install(FILES ${swig_generated_module} DESTINATION ${Python_SITELIB})

  endfunction()


  build_pressio_swig_module(pressio.i)
  build_pressio_swig_module(pressio_sz.i)
  build_pressio_swig_module(pressio_zfp.i)


endif()

