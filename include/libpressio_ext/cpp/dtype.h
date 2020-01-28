#include <pressio_dtype.h>

/**
 * Convert types to pressio_dtypes
 *
 * \tparam T the type to identify
 * \returns which pressio_dtype corresponds to the type T.
 */
template <class T>
constexpr pressio_dtype pressio_dtype_from_type() {
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

