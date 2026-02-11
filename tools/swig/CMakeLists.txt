cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
find_package(Python3 COMPONENTS Interpreter Development.Module)
find_package(SWIG COMPONENTS python REQUIRED)
find_package(dlpack)

if(NOT dlpack_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        dlpack
        GIT_REPOSITORY https://github.com/dmlc/dlpack
        GIT_TAG 93c8f2a3c774b84af6f652b1992c48164fae60fc #git version 1.2
       )
    FetchContent_MakeAvailable(dlpack)
endif()

if(SWIG_FOUND AND PYTHON3_FOUND)
  include(UseSWIG)


  set(CMAKE_SWIG_FLAGS "-doxygen")
  if(NOT LIBPRESSIO_PYTHON_SITELIB)
      if(DEFINED SKBUILD)
          set(LIBPRESSIO_PYTHON_SITELIB ${SKBUILD_PLATLIB_DIR} CACHE PATH "path to install python libraries to")
      else()
          set(LIBPRESSIO_PYTHON_SITELIB ${Python3_SITELIB} CACHE PATH "path to install python libraries to")
      endif()
  endif()
  set_property(SOURCE pressio.i PROPERTY CPLUSPLUS ON)
  set_property(SOURCE pressio.i PROPERTY USE_TARGET_INCLUDE_DIRECTORIES ON)
  set_property(SOURCE pressio.i PROPERTY SWIG_MODULE_NAME pressio)
  if(DEFINED SKBUILD)
      swig_add_library(
          pressio
          LANGUAGE python OUTPUT_DIR "${SKBUILD_PLATLIB_DIR}"
          SOURCES pressio.i
      )
      if(WIN32)
          set_property(TARGET pressio PROPERTY SUFFIX ".${SKBUILD_SOABI}.pyd")
      else()
          set_property(TARGET pressio
              PROPERTY SUFFIX ".${SKBUILD_SOABI}${CMAKE_SHARED_MODULE_SUFFIX}")
      endif()
      if(APPLE)
          set(origin_token "@loader_path")
      else()
          set(origin_token "$ORIGIN")
      endif()
      set_property(TARGET pressio PROPERTY INSTALL_RPATH
          "${origin_token}/"
      )
  else()
      swig_add_library(
          pressio
          LANGUAGE python
          SOURCES pressio.i
      )
  endif()
  set_property(TARGET pressio PROPERTY SWIG_COMPILE_DEFINITIONS "SWIGWORDSIZE64")
  target_link_libraries(pressio PUBLIC libpressio Python3::Module dlpack::dlpack)
  target_sources(pressio PRIVATE pypressio.cc)
  target_include_directories(pressio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
  if(LIBPRESSIO_HAS_MPI4PY)
      execute_process(
          COMMAND ${Python3_EXECUTABLE} -c "import mpi4py; print(mpi4py.get_include())" 
          OUTPUT_VARIABLE MPI4PY_SWIG_DIR 
          RESULT_VARIABLE MPI4PY_FOUND 
          OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(NOT ${MPI4PY_FOUND} EQUAL "0")
          message(FATAL_ERROR "Could not locate mpi4py")
      endif()
      target_include_directories(pressio PRIVATE ${MPI4PY_SWIG_DIR} ${MPI_CXX_INCLUDE_DIRS})
  endif()
  get_property(swig_generated_module TARGET pressio PROPERTY SWIG_SUPPORT_FILES)


  install(TARGETS pressio DESTINATION ${LIBPRESSIO_PYTHON_SITELIB})
  install(FILES ${swig_generated_module} DESTINATION ${LIBPRESSIO_PYTHON_SITELIB})



  install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/libpressio.py" DESTINATION ${LIBPRESSIO_PYTHON_SITELIB})
endif()

