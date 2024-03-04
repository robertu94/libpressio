#ifndef LIBPRESSIO_EXT_PYTHON
#define LIBPRESSIO_EXT_PYTHON
#include <memory>

/**
 * \file
 * \brief header to ensure that python is only initialized once by libpressio projects
 */

namespace libpressio { namespace python_launch {

/**
 * struct that manages pybind initialization for LibPressio
 */
struct libpressio_external_pybind_manager;

/**
 * Create or Retain the lifetime of the pybind global objects.  Must be called and
 * the returned pointer retained for the duration of use of embedded pybind
 *
 * \return a shared pointer that retains the lifetime of the pybind global settings
 */
std::shared_ptr<libpressio_external_pybind_manager> get_library();

} }

#endif /* end of include guard: LIBPRESSIO_EXT_PYTHON */
