#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/dtype.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/compressor.h"
#include <vector>
#include <cstdint>
#include <algorithm>
#include <std_compat/bit.h>
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

struct lp_cuda_array_interface {
    std::vector<uint64_t> shape;
    std::string typestr;
    intptr_t ptr;
    bool read_only;
    int version;
    int stream;
};

lp_cuda_array_interface io_data_to_cuda_array(struct pressio_data* data) {
    lp_cuda_array_interface ret;
    static std::map<pressio_dtype, std::string> dtypes {
        {pressio_float_dtype ,"f4"},
        {pressio_double_dtype,"f8"},
        {pressio_int8_dtype  ,"i1"},
        {pressio_int16_dtype ,"i2"},
        {pressio_int32_dtype ,"i4"},
        {pressio_int64_dtype ,"i8"},
        {pressio_uint8_dtype ,"u1"},
        {pressio_uint16_dtype,"u2"},
        {pressio_uint32_dtype,"u4"},
        {pressio_uint64_dtype,"u8"},
        {pressio_bool_dtype  ,"b1"},
        {pressio_byte_dtype  ,"i1"},
    };
    auto dims = data->dimensions();
    std::reverse(dims.begin(), dims.end());
    ret.shape = dims;
    ret.typestr = ((data->dtype() == pressio_byte_dtype)
                       ? "|"
                       : (compat::endian::native == compat::endian::little ? "<" : ">")) +
                  dtypes.at(data->dtype());
    ret.ptr = reinterpret_cast<intptr_t>(data->data());
    ret.read_only = false;
    ret.version = 3;
    ret.stream = 1;
    return ret;
}

pressio_data* io_data_from_cuda_array(std::vector<uint64_t> shape, std::string const& typestr, intptr_t ptr_asint) {
    static std::map<std::string, pressio_dtype> dtypes {
        {"f4", pressio_float_dtype},
        {"f8", pressio_double_dtype},
        {"i1", pressio_int8_dtype},
        {"i2", pressio_int16_dtype},
        {"i4", pressio_int32_dtype},
        {"i8", pressio_int64_dtype},
        {"u1", pressio_uint8_dtype},
        {"u2", pressio_uint16_dtype},
        {"u4", pressio_uint32_dtype},
        {"u8", pressio_uint64_dtype},
        {"b1", pressio_bool_dtype},
        {"i1"  ,pressio_byte_dtype},
    };
    pressio_dtype dtype;
    if(typestr.size() == 3) {
        auto endian = typestr[0];
        if (compat::endian::native == compat::endian::little) {
            if(endian == '>') {
                throw std::runtime_error("cross endian data not supported");
            }
        }
        dtype = dtypes.at(typestr.substr(1));
    }
    std::reverse(shape.begin(), shape.end());
    void* ptr = reinterpret_cast<void*>(ptr_asint);
    return new pressio_data(pressio_data::nonowning(dtype, ptr, shape, "cudamalloc"));
}

pressio_data* io_data_from_numpy_array(std::vector<uint64_t> shape, std::string const& typestr, intptr_t ptr_asint) {
    static std::map<std::string, pressio_dtype> dtypes {
        {"f4", pressio_float_dtype},
        {"f8", pressio_double_dtype},
        {"i1", pressio_int8_dtype},
        {"i2", pressio_int16_dtype},
        {"i4", pressio_int32_dtype},
        {"i8", pressio_int64_dtype},
        {"u1", pressio_uint8_dtype},
        {"u2", pressio_uint16_dtype},
        {"u4", pressio_uint32_dtype},
        {"u8", pressio_uint64_dtype},
        {"b1", pressio_bool_dtype},
        {"i1"  ,pressio_byte_dtype},
    };
    pressio_dtype dtype;
    if(typestr.size() == 3) {
        auto endian = typestr[0];
        if (compat::endian::native == compat::endian::little) {
            if(endian == '>') {
                throw std::runtime_error("cross endian data not supported");
            }
        }
        dtype = dtypes.at(typestr.substr(1));
    }
    std::reverse(shape.begin(), shape.end());
    void* ptr = reinterpret_cast<void*>(ptr_asint);
    return new pressio_data(pressio_data::nonowning(dtype, ptr, shape));
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

int compressor_compress_many(struct pressio_compressor* compressor, std::vector<struct pressio_data*> const& inputs, std::vector<struct pressio_data*>& outputs) {
    return (*compressor)->compress_many(inputs.begin(), inputs.end(), outputs.begin(), outputs.end());
}
int compressor_decompress_many(struct pressio_compressor* compressor, std::vector<struct pressio_data*> const& inputs, std::vector<struct pressio_data*>& outputs) {
    return (*compressor)->decompress_many(inputs.begin(), inputs.end(), outputs.begin(), outputs.end());
}

void options_set_strings(pressio_options* options, std::string const& key, std::vector<std::string> const& values){
    options->set(key, values);
}
