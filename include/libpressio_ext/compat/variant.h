/**
 * \file
 * \brief back ports of `<variant>`
 */
#ifndef LIBPRESSIO_COMPAT_VARIANT_H
#define LIBPRESSIO_COMPAT_VARIANT_H
#include <pressio_version.h>

#if !(LIBPRESSIO_COMPAT_HAS_VARIANT)
#include <boost/variant.hpp>
#else
#include <variant>
#endif


namespace compat {
#if (!LIBPRESSIO_COMPAT_HAS_VARIANT)
  using boost::variant;
  using boost::get;
  struct monostate {};
  template <typename T, typename... Ts>
  bool holds_alternative(const boost::variant<Ts...>& v) noexcept
  {
      return boost::get<T>(&v) != nullptr;
  }
#else
  using std::variant;
  using std::monostate;
  using std::holds_alternative;
  using std::get;
#endif

}

#endif /* end of include guard: LIBPRESSIO_COMPAT_VARIANT_H */
