/**
 * \file
 * \brief back ports of `<optional>`
 */
#ifndef LIBPRESSIO_COMPAT_OPTIONAL_H
#define LIBPRESSIO_COMPAT_OPTIONAL_H
#include <pressio_version.h>


#if !(LIBPRESSIO_COMPAT_HAS_OPTIONAL)
#include <boost/optional.hpp>
#include <boost/none.hpp>
#else
#include <optional>
#endif

namespace compat {
#if (!LIBPRESSIO_COMPAT_HAS_OPTIONAL)
  using boost::optional;
  extern const boost::none_t& nullopt;
#else
  using std::optional;
  using std::nullopt;
#endif
}


#endif /* end of include guard: LIBPRESSIO_COMPAT_OPTIONAL_H */
