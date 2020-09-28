/**
 * \file
 * \brief back ports of `<optional>`
 */
#ifndef LIBPRESSIO_COMPAT_STRING_VIEW_H
#define LIBPRESSIO_COMPAT_STRING_VIEW_H
#include <pressio_version.h>


#if !(LIBPRESSIO_COMPAT_HAS_STRING_VIEW)
#include <boost/utility/string_view.hpp>
#else
#include <string_view>
#endif

namespace compat {
#if (!LIBPRESSIO_COMPAT_HAS_STRING_VIEW)
  using boost::string_view;
#else
#include<string_view>
  using std::string_view;
#endif
}


#endif /* end of include guard: LIBPRESSIO_COMPAT_STRING_VIEW_H */
