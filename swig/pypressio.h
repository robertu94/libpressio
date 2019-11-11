#include "pressio_data.h"
#include "libpressio_ext/cpp/data.h"
#include <vector>
#include <cstdint>
#include <algorithm>

namespace {
  template <class T>
  constexpr pressio_dtype type_to_dtype() {
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

  template <class T>
  pressio_data*
  _pressio_io_data_from_numpy_impl(T* data, std::vector<size_t> dims) {
    return pressio_data_new_copy(
        type_to_dtype<T>(),
        data,
        dims.size(),
        dims.data()
        );
  }
}

template <class T>
pressio_data* 
_pressio_io_data_from_numpy_1d(T* data, size_t r1) {
  std::vector<size_t> v = {r1};
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
pressio_data* 
_pressio_io_data_from_numpy_2d(T* data, size_t r1, size_t r2) {
  std::vector<size_t> v = {r1, r2};
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
pressio_data* 
_pressio_io_data_from_numpy_3d(T* data, size_t r1, size_t r2, size_t r3) {
  std::vector<size_t> v = {r1, r2, r3};
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
pressio_data* 
_pressio_io_data_from_numpy_4d(T* data, size_t r1, size_t r2, size_t r3, size_t r4) {
  std::vector<size_t> v = {r1, r2, r3, r4};
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
std::vector<T> _pressio_io_data_to_numpy(pressio_data* ptr) {
  T* data = static_cast<T*>(pressio_data_ptr(ptr, nullptr));
  return std::vector<T>(
      data,
      data + pressio_data_num_elements(ptr)
      );
}


struct pressio_data* data_new_empty(const pressio_dtype dtype, std::vector<uint64_t> dimensions) {
  return new pressio_data(pressio_data::empty(dtype, dimensions));
}
struct pressio_data* data_new_nonowning(const pressio_dtype dtype, void* data, std::vector<uint64_t> dimensions) {
  return new pressio_data(pressio_data::nonowning(dtype, data, dimensions));
}
struct pressio_data* data_new_copy(const enum pressio_dtype dtype, void* src, std::vector<uint64_t>  dimensions) {
  return new pressio_data(pressio_data::copy(dtype, src, dimensions));
}
struct pressio_data* data_new_owning(const pressio_dtype dtype, std::vector<uint64_t> dimensions) {
  return new pressio_data(pressio_data::owning(dtype, dimensions));
}
struct pressio_data* data_new_move(const pressio_dtype dtype,
    void* data,
    std::vector<uint64_t> dimensions,
    pressio_data_delete_fn deleter,
    void* metadata) {
  return new pressio_data(pressio_data::move(dtype, data, dimensions, deleter, metadata));
}

pressio_metrics* new_metrics(struct pressio* library, std::vector<std::string> metrics) {
  std::vector<const char*> m;
  std::transform(std::begin(metrics), std::end(metrics), std::back_inserter(m), [](std::string& i){ return i.c_str(); });
  return pressio_new_metrics(library, m.data(), m.size());
}
