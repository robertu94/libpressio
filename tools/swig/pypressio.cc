#include "pypressio.h"
#include <std_compat/numeric.h>

namespace {
#if defined(_WIN32)
constexpr bool native_little_endian = true;
#else
constexpr bool native_little_endian = __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__;
#endif
}

void options_set_comm(struct pressio_options* options, const char* key, MPI_Comm comm) {
  MPI_Comm* c = new MPI_Comm(comm);
  return pressio_options_set_userptr_managed(options, key, c, nullptr, newdelete_deleter<MPI_Comm>(), newdelete_copy<MPI_Comm>());
}

pressio_option* option_new_comm(MPI_Comm comm) {
  MPI_Comm* c = new MPI_Comm(comm);
  return pressio_option_new_userptr_managed(c, nullptr, newdelete_deleter<MPI_Comm>(), newdelete_copy<MPI_Comm>());
}

static std::string dtype_to_typestr(pressio_dtype t) {
    static std::map<pressio_dtype, std::string> const dtypes {
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
    return ((t == pressio_byte_dtype)
                       ? "|"
                       : (native_little_endian ? "<" : ">")) +
                  dtypes.at(t);
}

static pressio_dtype dtype_from_typestr(std::string const& typestr) {
    static std::map<std::string, pressio_dtype> const dtypes {
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
    pressio_dtype dtype = pressio_byte_dtype;
    if(typestr.size() == 3) {
        auto endian = typestr[0];
        if (native_little_endian) {
            if(endian == '>') {
                throw std::runtime_error("cross endian data not supported");
            }
        }
        dtype = dtypes.at(typestr.substr(1));
    }
    return dtype;
}

numpy_array_interface data_to_array_interface(pressio_data* data) {
    numpy_array_interface i;
    i.shape = data->dimensions();
    std::reverse(i.shape.begin(), i.shape.end());
    i.typestr = dtype_to_typestr(data->dtype());
    i.read_only = false;
    i.ptr = reinterpret_cast<intptr_t>(data->data());
    i.version = 3;
    return i;
}


DLDevice data_to_dlpack_device(pressio_data* data) {
    DLDevice d;
    auto domain = data->domain();

    if(domain->domain_id() == "malloc") {
        d.device_type = kDLCPU;
        d.device_id = 0;
    } else if (domain->domain_id() == "cudamalloc") {
        d.device_type = kDLCUDA;
        d.device_id = 0; // TODO support multiple devices
    } else if (domain->domain_id() == "cudamallochost") {
        d.device_type = kDLCUDAHost;
        d.device_id = 0; // TODO support multiple devices
    }
    return d;
}

static DLDataType to_dlpack_dtype(pressio_dtype p) {
    DLDataType d;
    d.bits = pressio_dtype_size(p)*8;
    switch (p) {
        case pressio_float_dtype:
        case pressio_double_dtype:
            d.code = kDLFloat;
            break;
        case pressio_int8_dtype:
        case pressio_int16_dtype:
        case pressio_int32_dtype:
        case pressio_int64_dtype:
            d.code = kDLInt;
            break;
        case pressio_byte_dtype:
        case pressio_uint8_dtype:
        case pressio_uint16_dtype:
        case pressio_uint32_dtype:
        case pressio_uint64_dtype:
            d.code = kDLUInt;
            break;
        case pressio_bool_dtype:
            d.code = kDLBool;
            break;
    }
    d.lanes = 1;
    return d;
}

static DLTensor to_dltensor(pressio_data* data) {
    DLTensor d;
    d.byte_offset = 0;
    d.data = data->data();
    d.device = data_to_dlpack_device(data);
    d.dtype = to_dlpack_dtype(data->dtype());
    d.ndim = data->num_dimensions();

    auto dims = data->dimensions();
    d.shape = new int64_t[d.ndim];
    d.strides = new int64_t[d.ndim];
    std::copy(dims.rbegin(), dims.rend(), d.shape);
    compat::exclusive_scan(dims.rbegin(), dims.rend(), d.strides, 1, std::multiplies{});
    return d;
}

static void managed_deleter(DLManagedTensor *self) {
    delete[] self->dl_tensor.strides;
    delete[] self->dl_tensor.shape;
}

DLManagedTensor data_to_dlpack_managed(pressio_data* data) {
    DLManagedTensor d;
    d.dl_tensor = to_dltensor(data);
    d.deleter = managed_deleter;
    d.manager_ctx = nullptr;
    return d;
}

static void versioned_deleter(struct DLManagedTensorVersioned *self) {
    delete[] self->dl_tensor.strides;
    delete[] self->dl_tensor.shape;
}

DLManagedTensorVersioned data_to_dlpack_versioned(pressio_data* data, optional_max_version max_version, optional_dl_device dl_device, optional_copy copy) {
    DLManagedTensorVersioned d;
    d.version = DLPackVersion(DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION);
    d.manager_ctx = nullptr;
    d.flags = 0;
    d.deleter = versioned_deleter;
    d.dl_tensor = to_dltensor(data);
    return d;
}

std::vector<std::string> option_get_strings(pressio_option const* options) {
  return options->get_value<std::vector<std::string>>();
}

void option_set_strings(pressio_option* options, std::vector<std::string> const& strings) {
  *options = strings;
}

std::vector<uint64_t> data_dimensions(const pressio_data* data) {
    auto d =  data->dimensions();
    return std::vector<uint64_t>(d.begin(), d.end());
}

intptr_t data_ptr(const pressio_data* data) {
    return reinterpret_cast<intptr_t>(pressio_data_ptr(data, nullptr));
}

struct pressio_option* option_new_strings(std::vector<std::string> const& strings) {
  return new pressio_option(pressio_option(strings));
}
struct pressio_option* option_new_string(std::string const& string) {
  return new pressio_option(pressio_option(string));
}

struct pressio_data* data_new_empty(const pressio_dtype dtype, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  return new pressio_data(pressio_data::empty(dtype, dims));
}
struct pressio_data* data_new_nonowning(const pressio_dtype dtype, void* data, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  return new pressio_data(pressio_data::nonowning(dtype, data, dims));
}
struct pressio_data* data_new_nonowning_ptr(const pressio_dtype dtype, intptr_t data, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  return new pressio_data(pressio_data::nonowning(dtype, reinterpret_cast<void*>(data), dims));
}
struct pressio_data* data_new_copy(const enum pressio_dtype dtype, void* src, std::vector<uint64_t>  dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  return new pressio_data(pressio_data::copy(dtype, src, dims));
}
struct pressio_data* data_new_owning(const pressio_dtype dtype, std::vector<uint64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  return new pressio_data(pressio_data::owning(dtype, dims));
}
struct pressio_data* data_new_move(const pressio_dtype dtype,
    void* data,
    std::vector<uint64_t> dimensions,
    pressio_data_delete_fn deleter,
    void* metadata) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  return new pressio_data(pressio_data::move(dtype, data, dims, deleter, metadata));
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
