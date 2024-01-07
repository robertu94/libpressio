#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/dtype.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include "pressio_version.h"

#if LIBPRESSIO_HAS_MPI4PY
#include <mpi.h>

void options_set_comm(struct pressio_options* options, const char* key, MPI_Comm comm) {
  MPI_Comm* c = new MPI_Comm(comm);
  return pressio_options_set_userptr_managed(options, key, c, nullptr, newdelete_deleter<MPI_Comm>(), newdelete_copy<MPI_Comm>());
}

pressio_option* option_new_comm(MPI_Comm comm) {
  MPI_Comm* c = new MPI_Comm(comm);
  return pressio_option_new_userptr_managed(c, nullptr, newdelete_deleter<MPI_Comm>(), newdelete_copy<MPI_Comm>());
}

#endif

namespace {
  template <class T>
  pressio_data*
  _pressio_io_data_from_numpy_impl(T* data, std::vector<size_t> dims) {
    return pressio_data_new_copy(
        pressio_dtype_from_type<T>(),
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
  std::vector<size_t> v = {r2, r1}; //numpy presents dimension in C order, reverse them
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
pressio_data* 
_pressio_io_data_from_numpy_3d(T* data, size_t r1, size_t r2, size_t r3) {
  std::vector<size_t> v = {r3, r2, r1}; //numpy presents dimension in C order, reverse them
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
pressio_data* 
_pressio_io_data_from_numpy_4d(T* data, size_t r1, size_t r2, size_t r3, size_t r4) {
  std::vector<size_t> v = {r4, r3, r2, r1}; //numpy presents dimension in C order, reverse them
  return _pressio_io_data_from_numpy_impl(data, v);
}

template <class T>
void _pressio_io_data_to_numpy_1d(pressio_data* ptr, T** ptr_argout, long int*r1) {
  *r1 = ptr->get_dimension(0);
  *ptr_argout = static_cast<T*>(pressio_data_copy(ptr, nullptr));
}

template <class T>
void _pressio_io_data_to_numpy_2d(pressio_data* ptr, T** ptr_argout, long int*r1, long int* r2) {
  *r1 = ptr->get_dimension(1);
  *r2 = ptr->get_dimension(0);
  *ptr_argout = static_cast<T*>(pressio_data_copy(ptr, nullptr));
}

template <class T>
void _pressio_io_data_to_numpy_3d(pressio_data* ptr, T** ptr_argout, long int*r1, long int* r2, long int* r3) {
  *r1 = ptr->get_dimension(2);
  *r2 = ptr->get_dimension(1);
  *r3 = ptr->get_dimension(0);
  *ptr_argout = static_cast<T*>(pressio_data_copy(ptr, nullptr));
}

template <class T>
void _pressio_io_data_to_numpy_4d(pressio_data* ptr, T** ptr_argout, long int*r1, long int* r2, long int* r3, long int* r4) {
  *r1 = ptr->get_dimension(3);
  *r2 = ptr->get_dimension(2);
  *r3 = ptr->get_dimension(1);
  *r4 = ptr->get_dimension(0);
  *ptr_argout = static_cast<T*>(pressio_data_copy(ptr, nullptr));
}

std::string io_data_to_bytes(pressio_data* data) {
  size_t n_bytes;
  const char* bytes = static_cast<const char*>(pressio_data_ptr(data, &n_bytes));

  return std::string(bytes, n_bytes);
}

pressio_data* io_data_from_bytes(const char* buffer, size_t buffer_size) {
  return pressio_data_new_copy(pressio_byte_dtype, (void*)buffer, 1, &buffer_size);
}

std::vector<std::string> option_get_strings(pressio_option const* options) {
  return options->get_value<std::vector<std::string>>();
}

void option_set_strings(pressio_option* options, std::vector<std::string> const& strings) {
  *options = strings;
}

struct pressio_option* option_new_strings(std::vector<std::string> const& strings) {
  return new pressio_option(pressio_option(strings));
}

struct pressio_data* data_new_empty(const pressio_dtype dtype, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dimensions_(dimensions.cbegin(), dimensions.cend());
  return new pressio_data(pressio_data::empty(dtype, dimensions_));
}
struct pressio_data* data_new_nonowning(const pressio_dtype dtype, void* data, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dimensions_(dimensions.cbegin(), dimensions.cend());
  return new pressio_data(pressio_data::nonowning(dtype, data, dimensions_));
}
struct pressio_data* data_new_copy(const enum pressio_dtype dtype, void* src, std::vector<uint64_t>  dimensions) {
  std::vector<size_t> dimensions_(dimensions.cbegin(), dimensions.cend());
  return new pressio_data(pressio_data::copy(dtype, src, dimensions_));
}
struct pressio_data* data_new_owning(const pressio_dtype dtype, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dimensions_(dimensions.cbegin(), dimensions.cend());
  return new pressio_data(pressio_data::owning(dtype, dimensions_));
}
struct pressio_data* data_new_move(const pressio_dtype dtype,
    void* data,
    std::vector<uint64_t> dimensions,
    pressio_data_delete_fn deleter,
    void* metadata) {
  std::vector<size_t> dimensions_(dimensions.cbegin(), dimensions.cend());
  return new pressio_data(pressio_data::move(dtype, data, dimensions_, deleter, metadata));
}

pressio_metrics* new_metrics(struct pressio* library, std::vector<std::string> metrics) {
  std::vector<const char*> m;
  std::transform(std::begin(metrics), std::end(metrics), std::back_inserter(m), [](std::string& i){ return i.c_str(); });
  return pressio_new_metrics(library, m.data(), m.size());
}
