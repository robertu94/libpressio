cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(libpressio VERSION "0.99.6" LANGUAGES CXX C)

#correct was to set a default build type
# https://blog.kitware.com/cmake-and-the-default-build-type/
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "No build type was set. Setting build type to ${default_build_type}.")
  set(CMAKE_BUILD_TYPE ${default_build_type} CACHE 
    STRING "Choose the type to build" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")


include(CTest)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
option(BUILD_SHARED_LIBS "build libpressio as a shared library" ON)


include(GNUInstallDirs)
include(CheckCXXCompilerFlag)
include(CMakeDependentOption)


if(NOT USE_SANITIZER)
  set(USE_SANITIZER "" CACHE STRING "use a build sanitizer" FORCE)
endif()
set_property(CACHE USE_SANITIZER PROPERTY STRINGS "" "Address" "Thread" "Undefined" "Leak" "Memory")
if (USE_SANITIZER STREQUAL "Address")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=address")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address")
elseif (USE_SANITIZER STREQUAL "Thread")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=thread")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=thread")
elseif (USE_SANITIZER STREQUAL "Undefined")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=undefined")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=undefined")
elseif (USE_SANITIZER STREQUAL "Leak")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=leak")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=leak")
elseif (USE_SANITIZER STREQUAL "Memory")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fsanitize=memory")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=memory")
endif()

check_cxx_compiler_flag("-fno-omit-frame-pointer" HAVE_NO_OMIT_FRAME_POINTER)
set(NO_OMIT_FRAME_POINTER_FLAG "")
if(HAVE_NO_OMIT_FRAME_POINTER)
  set(NO_OMIT_FRAME_POINTER_FLAG "-fno-omit-frame-pointer")
endif()


set(LIBPRESSIO_FEATURES "")
set(LIBPRESSIO_COMPRESSORS "noop")

add_library(libpressio
  #core implementation
  ./src/pressio.cc
  ./src/userptr.cc
  ./src/pressio_posix.cc
  ./src/pressio_configurable.cc
  ./src/pressio_versionable.cc
  ./src/pressio_errorable.cc
  ./src/pressio_compressor.cc
  ./src/pressio_data.cc
  ./src/pressio_dtype.cc
  ./src/pressio_metrics.cc
  ./src/pressio_option.cc
  ./src/pressio_options.cc
  ./src/pressio_memory.cc
  ./src/pressio_domain.cc
  ./src/pressio_domain_send.cc
  ./src/pressio_domain_manager.cc
  ./src/pressio_options_iter.cc
  ./src/pressio_highlevel.cc
  ./src/external_parse.cc

  #plugins
  ./src/plugins/compressors/compressor_base.cc
  ./src/plugins/compressors/noop.cc
  ./src/plugins/compressors/pressio.cc
  ./src/plugins/metrics/composite.cc
  ./src/plugins/metrics/external.cc
  ./src/plugins/metrics/error_stat.cc
  ./src/plugins/metrics/metrics_base.cc
  ./src/plugins/launch/external_forkexec.cc
  ./src/plugins/launch_metrics/noop.cc
  ./src/plugins/launch_metrics/print.cc
  ./src/plugins/metrics/noop.cc
  ./src/plugins/io/noop.cc
  ./src/plugins/io/by_extension.cc
  ./src/plugins/io/io.cc
  ./src/plugins/io/posix.cc
  ./src/plugins/domains/base.cc
  ./src/plugins/domains/malloc.cc
  ./src/plugins/domains/nonowning.cc
  ./src/plugins/domains/user.cc
  ./src/iless.cc

  #public headers
  include/libpressio.h
  include/libpressio_ext/cpp/compressor.h
  include/libpressio_ext/cpp/configurable.h
  include/libpressio_ext/cpp/data.h
  include/libpressio_ext/cpp/errorable.h
  include/libpressio_ext/cpp/io.h
  include/libpressio_ext/cpp/libpressio.h
  include/libpressio_ext/cpp/metrics.h
  include/libpressio_ext/cpp/options.h
  include/libpressio_ext/cpp/pressio.h
  include/libpressio_ext/cpp/printers.h
  include/libpressio_ext/cpp/subgroup_manager.h
  include/libpressio_ext/cpp/versionable.h
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
  
  #private headers
  src/external_parse.h
  src/multi_dimensional_iterator.h
  src/cleanup.h
  )

if(NOT LIBPRESSIO_BUILD_MODE)
  set(LIBPRESSIO_BUILD_MODE FULL CACHE STRING "enable sets of plugins" FORCE)
  set_property(CACHE LIBPRESSIO_BUILD_MODE PROPERTY STRINGS "CORE" "FULL")
else()
  set(LIBPRESSIO_BUILD_MODE ${LIBPRESSIO_BUILD_MODE} CACHE STRING "enable sets of plugins" FORCE)
  set_property(CACHE LIBPRESSIO_BUILD_MODE PROPERTY STRINGS "CORE" "FULL")
endif()

#optional components created with libpressio_optional_component don't require dependencies
#they can be enabled in two (non-exclusive) ways:
# 1 by 1 using -DLIBPRESSIO_HAS_${NAME}=ON
# enabling sets of plugins organized by level LIBPRESSIO_BUILD_MODE=${level}
#    CORE -- build no plugins except noop's and the composite metric
#    FULL -- build all optional plugins that don't have additional dependencies
function(libpressio_optional_component name description file_names)
  string(TOUPPER "${name}" name_upper)
  cmake_dependent_option(LIBPRESSIO_HAS_${name_upper} ${description} OFF "LIBPRESSIO_BUILD_MODE STREQUAL CORE" OFF)
  if(LIBPRESSIO_HAS_${name_upper} OR LIBPRESSIO_BUILD_MODE STREQUAL FULL)
    foreach(file_name IN LISTS file_names)
    target_sources(libpressio
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/${file_name}
      )
    endforeach()
  endif()
endfunction()

libpressio_optional_component(cast "build the cast compressor plugin" /compressors/cast.cc)
libpressio_optional_component(threshold_small "build the threshold_small compressor plugin" /compressors/threshold_small.cc)
libpressio_optional_component(remove_background "build the remove_background compressor plugin" /compressors/remove_background.cc)
libpressio_optional_component(mask_interpolation "build the mask_interpolation compressor plugin" /compressors/mask_interpolation.cc)
libpressio_optional_component(max_error "build the max_error metric plugin" /metrics/max_error.cc)
libpressio_optional_component(repeat "build the repeat compressor plugin" /compressors/repeat.cc)
libpressio_optional_component(pw_rel "build the pw_rel compressor plugin" /compressors/pw_rel.cc)
libpressio_optional_component(csv "build the CSV io plugin" /io/csv.cc)
libpressio_optional_component(numpy "build the numpy io plugin" /io/numpy.cc)
libpressio_optional_component(copy_template "build the numpy io plugin" /io/copy_template.cc)
libpressio_optional_component(size "build the size metrics plugin" /metrics/size.cc)
libpressio_optional_component(time "build the time metrics plugin" /metrics/time.cc)
libpressio_optional_component(autocorr "build the autocorr metrics plugin" /metrics/autocorr.cc)
libpressio_optional_component(chunking "build the chunking compressor plugin" "/compressors/chunking.cc;/compressors/chunking_impl.cc")
libpressio_optional_component(delta_encoding "build the delta_encoding compressor plugin" /compressors/delta_encoding.cc)
libpressio_optional_component(diff_pdf "build the difference probability distribution function metric" /metrics/diff_pdf.cc)
libpressio_optional_component(empty "build the empty IO plugin" /io/empty.cc)
libpressio_optional_component(entropy "build the entropy metrics plugin" /metrics/entropy.cc)
libpressio_optional_component(historian "build the historian metrics plugin" /metrics/historian.cc)
libpressio_optional_component(input_stats "build the input_stats metrics plugin" /metrics/input_stats.cc)
libpressio_optional_component(iota "build the iota io plugin" /io/iota.cc)
libpressio_optional_component(kl_divergance "build the kl_divergance plugin" /metrics/kl_divergance.cc)
libpressio_optional_component(ks_test "build the KS test metrics plugin" /metrics/ks_test.cc)
libpressio_optional_component(kth_error "build the kth error metrics plugin" /metrics/kth_error.cc)
libpressio_optional_component(linear_quantizer "build the linear quantizer compressor plugin" /compressors/linear_quantizer.cc)
libpressio_optional_component(log_transform "build the log_transform compressor plugin" /compressors/log_transform.cc)
libpressio_optional_component(mask1d "build the mask1d metrics plugin" /metrics/mask1d.cc)
libpressio_optional_component(pearsons "build the pearsons metrics plugin" /metrics/pearsons.cc)
libpressio_optional_component(printer "build the printer metrics plugin" /metrics/printer.cc)
libpressio_optional_component(region_of_interest "build the region_of_interest metrics plugin" /metrics/region_of_interest.cc)
libpressio_optional_component(resize "build the resize compressor plugin" /compressors/resize.cc)
libpressio_optional_component(sampling "build the sampling compressors plugin" /compressors/sampling.cc)
libpressio_optional_component(select "build the io selection plugin" /io/select.cc)
libpressio_optional_component(switch "build the switch comprssor plugin" /compressors/switch.cc)
libpressio_optional_component(transpose "build the transpose compressor plugin" /compressors/transpose.cc)
libpressio_optional_component(print_options "print the options passed to set_options" /metrics/print_options.cc)
libpressio_optional_component(data_gap "compute the data_gap" /metrics/data_gap.cc)
libpressio_optional_component(clipping "compute the number of values that exceed an absolute error bound" /metrics/clipping.cc)
libpressio_optional_component(ssim "computes the SSIM" /metrics/ssim.cc)
libpressio_optional_component(ssim "computes the sobolev norm" /metrics/sobolevp2.cc)
libpressio_optional_component(ssim "computes the gradient length" /metrics/gradlength.cc)
libpressio_optional_component(write_debug_inputs "metric that writes output buffers for testing" /metrics/write_debug_inputs.cc)
set(roibin_sources /compressors/roibin.cc /compressors/binning.cc /compressors/masked_binning.cc)
libpressio_optional_component(roibin "build the roibin metacompressor" "${roibin_sources}")
libpressio_optional_component(pipeline "build the pipeline metacompressor" /compressors/pipeline.cc)

option(LIBPRESSIO_INTERPROCEDURAL_OPTIMIZATION "Use interprocedural optimization (LTO)" OFF)
if(LIBPRESSIO_INTERPROCEDURAL_OPTIMIZATION)
  cmake_policy(CMP0069 NEW)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERR_MSG LANGUAGES CXX)
  if(IPO_SUPPORTED)
    set_target_properties(libpressio PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(ERROR "IPO was not supported: ${IPO_ERR_MSG}")
  endif()
endif()

target_include_directories(libpressio
  PRIVATE 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src> 
  )
target_include_directories(libpressio
  PUBLIC 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include> 
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/libpressio>
  )
target_compile_options(libpressio PRIVATE 
  $<$<COMPILE_LANGUAGE:CXX,C>:
      $<$<CONFIG:Debug>: -Wall -Wextra -Wpedantic>
      $<$<CONFIG:RelWithDebInfo>: ${NO_OMIT_FRAME_POINTER_FLAG}>
  >
  )

option(LIBPRESSIO_TREAT_WARNINGS_AS_ERRORS "treat compiler warnings as errors" OFF)
if(LIBPRESSIO_TREAT_WARNINGS_AS_ERRORS)
  target_compile_options(libpressio PRIVATE -Werror)
endif()

find_package(std_compat REQUIRED)
target_link_libraries(libpressio PUBLIC std_compat::std_compat)

option(LIBPRESSIO_HAS_OPENMP "accerate some plugins with OpenMP" OFF)
if(LIBPRESSIO_HAS_OPENMP)
  find_package(OpenMP)
  target_link_libraries(libpressio PRIVATE OpenMP::OpenMP_CXX)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/metrics/spatial_error.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/many_independent_threaded.cc
    )
endif()

option(LIBPRESSIO_HAS_LINUX "plugins that depend on various unixisms" OFF)
if(LIBPRESSIO_HAS_LINUX)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/metrics/rusage.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/mmap.cc
    )
endif()

option(LIBPRESSIO_HAS_ARC "build the ARC plugin" OFF)
if(LIBPRESSIO_HAS_ARC)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} ARC")
  find_package(ARC REQUIRED)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/arc.cc
    )
  target_link_libraries(libpressio PRIVATE ARC::arc)
endif()

option(LIBPRESSIO_HAS_DIGIT_ROUNDING "build the DIGIT ROUNDING plugin" OFF)
if(LIBPRESSIO_HAS_DIGIT_ROUNDING)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} DR")
  find_package(dround REQUIRED)
  find_package(ZLIB REQUIRED)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/digit_rounding_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE dround ZLIB::ZLIB)
endif()

option(LIBPRESSIO_HAS_BZIP2 "build the BZIP plugin" OFF)
if(LIBPRESSIO_HAS_BZIP2)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} BZIP2")
  find_package(BZip2 REQUIRED)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/bzip2.cc
    )
  target_link_libraries(libpressio PRIVATE BZip2::BZip2)
endif()

option(LIBPRESSIO_HAS_BIT_GROOMING "build the BIT GROOMING plugin" OFF)
if(LIBPRESSIO_HAS_BIT_GROOMING)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} BG")
  find_package(bg REQUIRED)
  find_package(ZLIB REQUIRED)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/bit_groooming_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE bg ZLIB::ZLIB)
endif()

option(LIBPRESSIO_HAS_SZ_AUTO "build the SZauto plugin" OFF)
if(LIBPRESSIO_HAS_SZ_AUTO)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} SZauto")
  find_package(szauto REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_auto_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE szauto::sz_cpp)
endif()


option(LIBPRESSIO_HAS_SZx "build the SZx plugin" OFF)
cmake_dependent_option(LIBPRESSIO_HAS_CUSZx "build cuda support for SZx" OFF "LIBPRESSIO_HAS_SZx" OFF)
if(LIBPRESSIO_HAS_SZx)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} SZx")
  find_package(SZx REQUIRED)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/szx.cc
    )
    target_link_libraries(libpressio PRIVATE SZx::SZx)
endif()


option(LIBPRESSIO_HAS_MGARDx "build the MGARDx plugin" OFF)
if(LIBPRESSIO_HAS_MGARDx)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} MGARDx")
  find_package(MGARDx)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/MGARDx_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE MGARDx::MGARDx)
endif()

option(LIBPRESSIO_HAS_MGARD "build the MGARD plugin" OFF)
if(LIBPRESSIO_HAS_MGARD)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} mgard")
  find_package(mgard REQUIRED)
  if(DEFINED mgard_VERSION_MAJOR AND (${mgard_VERSION_MAJOR} EQUAL 1 AND ${mgard_VERSION_MINOR} GREATER_EQUAL 3) OR ${mgard_VERSION_MAJOR} GREATER_EQUAL 2)
      target_sources(libpressio
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/mgard-1.3_plugin.cc
        )
      target_link_libraries(libpressio PRIVATE mgard::mgard)
    elseif(DEFINED mgard_VERSION_MAJOR AND ${mgard_VERSION_MAJOR} EQUAL 1 )
      target_sources(libpressio
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/mgard-1.0_plugin.cc
        )
      target_link_libraries(libpressio PRIVATE mgard::mgard)
  else()
    find_file(LIBPRESSIO_MGARD_NEED_FLOAT_HEADER mgard_api_float.h)
    find_file(LIBPRESSIO_MGARD_HAS_CONFIG_HEADER MGARDConfig.h)
    find_file(LIBPRESSIO_MGARD_HAS_CONFIG_HEADER_HPP MGARDConfig.hpp PATH_SUFFIXES mgard)
    target_sources(libpressio
      PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/mgard_plugin.cc
      )
    if(TARGET mgard::mgard)
      target_link_libraries(libpressio PRIVATE mgard::mgard)
    elseif(TARGET mgard::mgard_library)
      target_link_libraries(libpressio PRIVATE mgard::mgard_library)
    endif()
   endif()
endif()

option(LIBPRESSIO_HAS_ZFP "build the ZFP plugin" OFF)
if(LIBPRESSIO_HAS_ZFP)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} zfp")
  find_package(ZFP REQUIRED)
  try_compile(LIBPRESSIO_ZFP_HAS_HIP ${CMAKE_CURRENT_BINARY_DIR} SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/checks/zfp_hip.cc LINK_LIBRARIES zfp::zfp)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/zfp_plugin.cc
    )
  target_link_libraries(libpressio PRIVATE zfp::zfp)
endif()

option(LIBPRESSIO_HAS_CUSZ "build the CUSz plugin" OFF)
if(LIBPRESSIO_HAS_CUSZ)
  enable_language(CUDA)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} cusz")
  find_package(CUSZ REQUIRED)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/cusz.cc
  )
  target_link_libraries(libpressio PRIVATE CUSZ::cusz CUDA::cuda_driver CUDA::cudart)
endif()

option(LIBPRESSIO_HAS_SZ3 "build the SZ3 plugin" OFF)
if(LIBPRESSIO_HAS_SZ3)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} sz3")
  find_package(SZ3 REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz3.cc
  )
target_link_libraries(libpressio PRIVATE SZ3::SZ3)
endif()

option(LIBPRESSIO_HAS_QoZ "build the QoZ plugin" OFF)
if(LIBPRESSIO_HAS_QoZ)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} qoz")
  find_package(QoZ REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/qoz.cc
  )
target_link_libraries(libpressio PRIVATE QoZ::QoZ)
endif()


option(LIBPRESSIO_HAS_SZ "build the SZ plugin" OFF)
if(LIBPRESSIO_HAS_SZ)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} sz")
  find_package(SZ REQUIRED)
  find_package(ZLIB REQUIRED)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(ZSTD IMPORTED_TARGET GLOBAL libzstd)
  pkg_search_module(FFTW3 QUIET IMPORTED_TARGET GLOBAL fftw3)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_plugin.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_threadsafe_plugin.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_common.cc
  )
  target_link_libraries(libpressio PRIVATE SZ)
  target_include_directories(libpressio PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors)

  if(LIBPRESSIO_HAS_OPENMP)
    target_sources(libpressio
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/sz_omp.cc
    )
  endif()
endif()

option(LIBPRESSIO_HAS_BLOSC "build the BLOSC plugin" OFF)
if(LIBPRESSIO_HAS_BLOSC)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} blosc")
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/blosc_plugin.cc
    )
  find_package(PkgConfig REQUIRED)
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

option(LIBPRESSIO_HAS_BLOSC2 "build the BLOSC2 plugin" OFF)
if(LIBPRESSIO_HAS_BLOSC2)

    target_sources(libpressio
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/blosc2_plugin.cc
    )
    find_package(Blosc2 QUIET)
    if(Blosc2_FOUND)
        target_link_libraries(libpressio PRIVATE Blosc2::blosc2_shared)
    else()
        find_package(PkgConfig REQUIRED)
        pkg_search_module(Blosc2 IMPORTED_TARGET GLOBAL blosc2)
        target_link_libraries(libpressio PRIVATE PkgConfig::Blosc2)
    endif()
endif()

option(LIBPRESSIO_HAS_MAGICK "build the image magick plugin" OFF)
if(LIBPRESSIO_HAS_MAGICK)
  find_package(PkgConfig REQUIRED)
  pkg_search_module(Magick++ IMPORTED_TARGET GLOBAL Magick++)
  try_compile(
    LIBPRESSIO_COMPAT_HAS_IMAGEMAGICK_LONGLONG
    ${CMAKE_BINARY_DIR}
    SOURCES "${CMAKE_SOURCE_DIR}/checks/imagemagick_longlong.cc"
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

option(LIBPRESSIO_HAS_NETCDF "build the netcdf io plugin" OFF)
if(LIBPRESSIO_HAS_NETCDF)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} netcdf")
  find_package(PkgConfig REQUIRED)
  pkg_search_module(NetCDF REQUIRED IMPORTED_TARGET GLOBAL netcdf)
  target_link_libraries(libpressio PRIVATE PkgConfig::NetCDF)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/netcdf.cc
    )
endif()

option(LIBPRESSIO_HAS_CUDA "build the cuda_device_selector compressor plugin and cuda integration" OFF)
if(LIBPRESSIO_HAS_CUDA)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} cuda")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(libpressio PRIVATE CUDA::cuda_driver CUDA::cudart)
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/cuda_device_selector.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/domains/cuda.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/domains/cudahost.cc
    )
endif()


option(LIBPRESSIO_HAS_CUFILE "build the cufile io plugin" OFF)
if(LIBPRESSIO_HAS_CUFILE)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} cufile")
  find_package(CUDAToolkit REQUIRED)
  find_library(CUFile_LIBRARY cufile PATHS ${CUDAToolkit_LIBRARY_DIR} REQUIRED)
  target_link_libraries(libpressio PRIVATE CUDA::cuda_driver CUDA::cudart ${CUFile_LIBRARY})
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/cufile.cc
    )
endif()

option(LIBPRESSIO_HAS_MATLABIO "build the matlab io plugin" OFF)
if(LIBPRESSIO_HAS_MATLABIO)
    find_file(MATIO_HEADER matio.h REQUIRED)
    find_library(MATIO_LIBRARY matio REQUIRED)
    get_filename_component(MATIO_HEADER_DIR ${MATIO_HEADER} DIRECTORY)
    target_link_libraries(libpressio PRIVATE ${MATIO_LIBRARY})
    target_include_directories(libpressio PRIVATE ${MATIO_HEADER_DIR})
    target_sources(libpressio
      PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/matio.cc
      )
endif()

option(LIBPRESSIO_HAS_HDF "build the hdf5 io plugin" OFF)
if(LIBPRESSIO_HAS_HDF)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} hdf5")
  find_package(HDF5 REQUIRED COMPONENTS C)
  target_link_libraries(libpressio PRIVATE ${HDF5_C_LIBRARIES})
  target_include_directories(libpressio PRIVATE ${HDF5_C_INCLUDE_DIRS})
  target_compile_definitions(libpressio PRIVATE ${HDF5_C_DEFINITIONS})
  if(${HDF5_IS_PARALLEL})
    find_package(MPI REQUIRED)
    target_link_libraries(libpressio PRIVATE MPI::MPI_CXX)
  endif()
  target_sources(libpressio
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/hdf5.cc
      ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/io/hdf5.h
    )
endif()

option(LIBPRESSIO_HAS_NDZIP "build the ndzip plugin" OFF)
if(LIBPRESSIO_HAS_NDZIP)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} ndzip")
  find_package(NDZip REQUIRED)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/ndzip.cc
    )
  target_link_libraries(libpressio PRIVATE NDZip::ndzip)
  if(TARGET NDZip::ndzip-cuda)
    target_link_libraries(libpressio PRIVATE NDZip::ndzip-cuda)
  endif()
  if(TARGET NDZip::ndzip-sycl)
    target_link_libraries(libpressio PRIVATE NDZip::ndzip-sycl)
  endif()

endif()

option(LIBPRESSIO_HAS_CUSZP "build the CUSZP plugin" OFF)
if(LIBPRESSIO_HAS_CUSZP)
  set(LIBPRESSIO_COMPRESSORS "${LIBPRESSIO_COMPRESSORS} CUSZP")
  find_package(cuSZp REQUIRED)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/cuszp.cc
    )
  target_link_libraries(libpressio PRIVATE cuSZp::cuSZp)
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
cmake_dependent_option(LIBPRESSIO_HAS_MPI4PY "build mpi support for the python binding" OFF "LIBPRESSIO_HAS_MPI" OFF)
if(LIBPRESSIO_HAS_MPI)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} mpi")
  find_package(MPI COMPONENTS CXX)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/launch/external_mpispawn.cc
    )
  target_link_libraries(libpressio PRIVATE MPI::MPI_CXX)
endif()


option(LIBPRESSIO_HAS_PYTHON_LAUNCH "build the python launch plugin" OFF)
if(LIBPRESSIO_HAS_PYTHON_LAUNCH)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} python")
  find_package(pybind11 REQUIRED)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/launch/external_python.cc
    )
  target_link_libraries(libpressio PRIVATE pybind11::embed)
endif()

option(LIBPRESSIO_HAS_PETSC "build the petsc io plugin" OFF)
if(LIBPRESSIO_HAS_PETSC)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} petsc")
  find_package(PkgConfig REQUIRED)
  pkg_search_module(PETSc IMPORTED_TARGET PETSc)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/io/petsc.cc
    )
  target_link_libraries(libpressio PRIVATE PkgConfig::PETSc)
endif()

option(LIBPRESSIO_LUA_PKGCONFIG "prefer pkg-config to find lua" ON)
option(LIBPRESSIO_HAS_LUA "build multi-objective code" OFF)
if(LIBPRESSIO_HAS_LUA)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} lua")
  find_package(sol2 REQUIRED)
  if(LIBPRESSIO_LUA_PKGCONFIG)
      find_package(PkgConfig REQUIRED)
      pkg_search_module(Lua IMPORTED_TARGET lua luajit)
  else()
      set(Lua_FOUND FALSE)
  endif()
  if(Lua_FOUND)
    message(INFO "lua from pkgconfig")
    target_link_libraries(libpressio PRIVATE PkgConfig::Lua)
  else()
    message(INFO "lua from cmake")
    find_package(Lua)
    target_link_libraries(libpressio PRIVATE ${LUA_LIBRARIES})
    target_include_directories(libpressio PRIVATE ${LUA_INCLUDE_DIR})
  endif()
  target_sources(libpressio PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/lambda_fn.cc
    )
endif()

option(LIBPRESSIO_HAS_LIBDISTRIBUTED "build plugins based on distributed work queues" OFF)
if(LIBPRESSIO_HAS_LIBDISTRIBUTED)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} distributed")
  find_package(LibDistributed)
  target_link_libraries(libpressio PRIVATE LibDistributed::libdistributed)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/many_dependent.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/many_independent.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/serializable.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pressio_distributed_manager.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/cpp/distributed_manager.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/cpp/serializable.h

    )
endif()

option(LIBPRESSIO_HAS_JSON "build the options json serialization support" OFF)
if(LIBPRESSIO_HAS_JSON)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} json")
  find_package(nlohmann_json REQUIRED)
  target_link_libraries(libpressio PRIVATE nlohmann_json::nlohmann_json)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pressio_options_json.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/manifest.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/compressors/external.cc
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/json/pressio_options_json.h
    ${CMAKE_CURRENT_SOURCE_DIR}/include/libpressio_ext/cpp/json.h
    )

endif()

option(LIBPRESSIO_HAS_REMOTELAUNCH "build the remote external launch plugin" OFF)
if(LIBPRESSIO_HAS_REMOTELAUNCH)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} remotelaunch")
  find_package(nlohmann_json REQUIRED)
  find_package(CURL REQUIRED)
  target_link_libraries(libpressio PRIVATE nlohmann_json::nlohmann_json CURL::libcurl)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/plugins/launch/external_remotelaunch.cc
    )

endif()

option(LIBPRESSIO_HAS_OPENSSL "build with OpenSSL support to hash libpressio options" OFF)
if(LIBPRESSIO_HAS_OPENSSL)
  set(LIBPRESSIO_FEATURES "${LIBPRESSIO_FEATURES} remotelaunch")
  find_package(OpenSSL REQUIRED)
  target_link_libraries(libpressio PRIVATE OpenSSL::Crypto)
  target_sources(libpressio
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/pressio_hash.cc
    )
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
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio_cxx.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio_cxx.pc
  @ONLY
  )
if(LIBPRESSIO_HAS_HDF)
  configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/libpressio_hdf5.pc.in
  ${CMAKE_CURRENT_BINARY_DIR}/libpressio_hdf5.pc
  @ONLY
  )
endif()
target_sources(libpressio
  PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/include/pressio_version.h
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


install(TARGETS libpressio EXPORT LibPressioTargets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  )
install(EXPORT LibPressioTargets NAMESPACE LibPressio:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressio/)
export(EXPORT LibPressioTargets FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/LibPressioTargets.cmake NAMESPACE LibPressio::)
include(CMakePackageConfigHelpers)
configure_package_config_file(${CMAKE_CURRENT_SOURCE_DIR}/LibPressioConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioConfig.cmake"
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressio
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/LibPressioConfigVersion.cmake"
  VERSION "${PROJECT_VERSION}"
  COMPATIBILITY AnyNewerVersion
)
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/LibPressioConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/LibPressioConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LibPressio
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/include/pressio_version.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/libpressio)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio_cxx.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
if(LIBPRESSIO_HAS_HDF)
  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/libpressio_hdf5.pc DESTINATION ${CMAKE_INSTALL_PREFIX}/share/pkgconfig)
endif()

if(BUILD_TESTING)
  add_subdirectory(test)
endif()

option(BUILD_PYTHON_WRAPPER "build python wrapper" OFF)
if(BUILD_PYTHON_WRAPPER)
  add_subdirectory(tools/swig)
endif()

if(LIBPRESSIO_HAS_JSON AND LIBPRESSIO_HAS_HDF)
  add_subdirectory(tools/hdf5_filter)
endif()

option(BUILD_MIGRATION_TOOLS "build tools to help migrate to newer versions of libpressio" OFF)
if(BUILD_MIGRATION_TOOLS)
    add_subdirectory(tools/migrate/)
endif()

option(BUILD_DOCS "build the documetation" OFF)
if(BUILD_DOCS)
  add_subdirectory(tools/docs)


  option(BUILD_DOCS_LINKS "build the documetation" ON)
  find_package(Doxygen REQUIRED dot)
  if(BUILD_DOCS_LINKS)
    set(DOXYGEN_MAN_LINKS YES)
  else()
    set(DOXYGEN_MAN_LINKS NO)
  endif()
  set(DOXYGEN_GENERATE_MAN YES)

  set(DOXYGEN_HTML_EXTRA_STYLESHEET "${CMAKE_CURRENT_SOURCE_DIR}/docs/doxygen-awesome.css" )
  set(DOXYGEN_HTML_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/docs/head.html")
  set(DOXYGEN_GENERATE_HTML YES)
  set(DOXYGEN_EXTRACT_LOCAL_METHODS YES)
  set(DOXYGEN_EXTRACT_STATIC YES)
  set(DOXYGEN_MACRO_EXPANSION YES)
  set(DOXYGEN_USE_MDFILE_AS_MAINPAGE README.md)
  option(LIBPRESSIO_INSTALL_DOCS "build the documetation by default" OFF)
  if(LIBPRESSIO_INSTALL_DOCS)
    set(LIBPRESSIO_BUILD_DOCS_DEFAULT "ALL")
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/html/" DESTINATION "${CMAKE_INSTALL_DOCDIR}")
    install(DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/man/" DESTINATION "${CMAKE_INSTALL_MANDIR}")
  else()
    set(LIBPRESSIO_BUILD_DOCS_DEFAULT "")
  endif()
  add_custom_command(
    OUTPUT "IO.md"
    COMMAND generate_docs -m i -o "${CMAKE_BINARY_DIR}/IO.md"
    DEPENDS generate_docs libpressio
    COMMENT "Documenting IO"
    )
  add_custom_command(
    OUTPUT "Compressors.md"
    COMMAND generate_docs -m c -o "${CMAKE_BINARY_DIR}/Compressors.md"
    DEPENDS generate_docs libpressio
    COMMENT "Documenting Compressors"
    )
  add_custom_command(
    OUTPUT "Metrics.md"
    COMMAND generate_docs -m m -o "${CMAKE_BINARY_DIR}/Metrics.md"
    DEPENDS generate_docs libpressio
    COMMENT "Documenting Metrics"
    )
  doxygen_add_docs(
    docs
    ${PROJECT_SOURCE_DIR}/docs/
    ${PROJECT_SOURCE_DIR}/README.md
    ${PROJECT_SOURCE_DIR}/CONTRIBUTORS.md
    ${PROJECT_BINARY_DIR}/Compressors.md
    ${PROJECT_BINARY_DIR}/Metrics.md
    ${PROJECT_BINARY_DIR}/IO.md
    ${PROJECT_SOURCE_DIR}/include
    ${LIBPRESSIO_BUILD_DOCS_DEFAULT}
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
