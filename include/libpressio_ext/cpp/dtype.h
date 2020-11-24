#ifndef LIBPRESSIO_DTYPE_CPP
#define LIBPRESSIO_DTYPE_CPP
#include <pressio_dtype.h>
#include <std_compat/type_traits.h>
#include <cstdint>
#include <stdint.h>

/**
 * \file
 * \brief C++ interface to data types
 */

namespace impl {
  template <class T, class... Ts>
  struct is_one_of : public compat::disjunction<std::is_same<T,Ts>...> {
  };
}


/**
 * Convert types to pressio_dtypes
 *
 * \tparam T the type to identify
 * \returns which pressio_dtype corresponds to the type T.
 */
template <class T>
constexpr pressio_dtype pressio_dtype_from_type() {
  static_assert(impl::is_one_of<T,double, float, int8_t, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>::value, 
      "unexpected type");
  return (std::is_same<T, double>::value ? pressio_double_dtype :
      std::is_same<T, float>::value ? pressio_float_dtype :
      std::is_same<T, int64_t>::value ? pressio_int64_dtype :
      std::is_same<T, int32_t>::value ? pressio_int32_dtype :
      std::is_same<T, int16_t>::value ? pressio_int16_dtype :
      std::is_same<T, int8_t>::value ? pressio_int8_dtype :
      std::is_same<T, uint64_t>::value ? pressio_uint64_dtype :
      std::is_same<T, uint32_t>::value ? pressio_uint32_dtype :
      std::is_same<T, uint16_t>::value ? pressio_uint16_dtype :
      std::is_same<T, uint8_t>::value ? pressio_uint8_dtype :
      pressio_byte_dtype
      );
}

#endif /* end of include guard: LIBPRESSIO_DTYPE_CPP */
