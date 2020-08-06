/**
 * \file
 * \brief back ports of `<cstddef>`
 */
#ifndef LIBPRESSIO_COMPAT_BYTE_H
#define LIBPRESSIO_COMPAT_BYTE_H
#include <pressio_version.h>
#include <cstddef>

namespace compat {
#if !LIBPRESSIO_COMPAT_HAS_BYTE
  /**
   * represents a byte unlike unsigned char does not provide arithmetic operators
   */
  enum class byte : unsigned char {};
#else
  using std::byte;
#endif
}

#endif /* end of include guard: LIBPRESSIO_COMPAT_BYTE_H */
