#include "libpressio_hdf5_filter.h"
#include "libpressio_hdf5_filter_impl.h"
#include <std_compat/span.h>
#include <libpressio_ext/cpp/libpressio.h>
#include <libpressio_ext/cpp/json.h>
#include <nlohmann/json.hpp>
#include <vector>
#include <string>
#include <endian.h>

compression_options get_options_from_cd_values(size_t cd_nelmts, const unsigned int* cd_values) {
  compression_options options;
  size_t current = 0;
  compat::span<const unsigned int> vec(cd_values, cd_nelmts);
  auto pop_word = [&current, &vec]() {
    const uint64_t* ret = reinterpret_cast<const uint64_t*>(&vec[current]);
    current += 2;
    return *ret;
  };
  options.dtype = pressio_dtype(vec[current++]);
  options.dims.resize(pop_word());
  for (size_t i = 0; i < options.dims.size(); ++i) {
    options.dims[i] = pop_word();
  }

  options.compressor_id.resize(pop_word());
  memcpy(&options.compressor_id[0], &vec[current], options.compressor_id.size());
  current += ((options.compressor_id.size() /4) + 1);

  std::string op_str(pop_word(), 0);
  memcpy(&op_str[0], &vec[current], op_str.size());
  nlohmann::json j = nlohmann::json::from_msgpack(op_str);
  options.options = j;
  
  return options;
}

std::vector<unsigned int> get_cd_values_from_options(compression_options const& options) {
  std::vector<unsigned int> ret;
  auto push_word = [&ret](uint64_t value) {
    ret.push_back(value & 0x0000FFFF);
    ret.push_back(value >> 32);
  };
  auto push_bytes = [&ret, &push_word](const char* bytes, size_t len) {
    push_word(len);
    const size_t current_size = ret.size();
    ret.resize(current_size + (len/4) +1);
    memcpy(&ret[current_size], bytes, len);
  };

  ret.push_back(options.dtype);
  push_word(options.dims.size());
  for (auto i : options.dims) {
    push_word(i);
  }

  push_bytes(options.compressor_id.c_str(), options.compressor_id.size());

  nlohmann::json j = options.options;
  std::vector<uint8_t> msgpk = nlohmann::json::to_msgpack(j);
  push_bytes(reinterpret_cast<const char*>(msgpk.data()), msgpk.size());

  return ret;
}

extern "C" {

#define H5Z_LIBPRESSIO_PUSH_AND_GOTO(MAJ, MIN, RET, MSG) \
  do {                                                                         \
    H5Epush(H5E_DEFAULT, __FILE__, _funcname_, __LINE__, H5E_ERR_CLS, MAJ,     \
            MIN, MSG);                                                         \
    retval = RET;                                                              \
    goto done;                                                                 \
  } while (0)


  /**
   * apply a libpressio compressor to an HDF5 chunk
   *
   * \param[in] flags - options passed to the HDF5 filter plugin such as
   * H5Z_FLAG_REVERSE which indicates decompression
   * \param[in] cd_nelmts - the number of auxiliary values passed to the filter
   * \param[in] cd_values - auxiliary values passed to the filter
   * \param[in] n_bytes - the size of the input buffer in bytes 
   * \param[out] buf_size - the size of the output buffer in bytes 
   * \param[in,out] buf - on input, the actual memory to be compressed/decompressed; on output, the
   *            compressed/decompressed memory; we can't assume the operation can be done
   *            in-place, so buf needs to be freed before re-assigned
   * \return 
   */
  static size_t H5Z_filter_libpressio(unsigned int flags, size_t cd_nelmts,
                                      const unsigned int cd_values[],
                                      size_t n_bytes, size_t* buf_size,
                                      void** buf)
  {
    //parse compression_options from cd_values, cd_nelmts
    auto comp_options = get_options_from_cd_values(cd_nelmts, cd_values);
    
    //instantiate the data

    //instantiate the compressor from cd_values
    pressio library;
    pressio_compressor compressor = library.get_compressor(comp_options.compressor_id);
    compressor->set_options(comp_options.options);
        
    if (flags & H5Z_FLAG_REVERSE) {
      // preform decompression
      pressio_data const& input = pressio_data::nonowning(pressio_byte_dtype, *buf, {n_bytes});
      pressio_data output = pressio_data::owning(comp_options.dtype, comp_options.dims);
      int rc = compressor->decompress(&input, &output);
      if(rc == 0) {
        H5free_memory(*buf);
        *buf = pressio_data_copy(&output, buf_size);
      } else {
        *buf_size = 0;
      }
    } else {
      // preform compression
      pressio_data const& input = pressio_data::nonowning(comp_options.dtype, *buf, comp_options.dims);
      pressio_data output = pressio_data::empty(pressio_byte_dtype, {});
      int rc = compressor->compress(&input, &output);
      if(rc == 0) {
        H5free_memory(*buf);
        *buf = pressio_data_copy(&output, buf_size);
      } else {
        *buf_size = 0;
      }
    }

    return *buf_size;
  }

  /**
   * apply local options to this libpressio compressor
   *
   * \param[in] dcpl_id  the dataset creation property list
   * \param[in] type_id  datatype identifier passed in to H5Dcreate. should not be modified 
   * \param[in] chunk_space_id  a dataspace describing the chunk.  should not be modified
   * \returns non-negative value on success and a negative value for an error
   */
  static herr_t H5Z_libpressio_set_local(hid_t dcpl_id, hid_t type_id,
                                         hid_t chunk_space_id)
  {
    static char const*_funcname_ = "H5Z_libpressio_can_apply";
    compression_options opts;
    const int h_ndims = H5Sget_simple_extent_ndims(chunk_space_id);
    const hid_t native_type = H5Tget_native_type(type_id, H5T_DIR_DEFAULT);
    std::vector<hsize_t> h_dims;
    std::vector<unsigned int> cd_vals;
    unsigned int flags;
    herr_t retval = 1;
    pressio_options* p_opts = nullptr;
    std::string* p_string = nullptr;

    if(h_ndims < 0) {
      H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a data space");
    } else {
      h_dims.resize(h_ndims);
    }

    if(H5Sget_simple_extent_dims(chunk_space_id, h_dims.data(), nullptr) > 0) {
      opts.dims = std::vector<size_t>(h_dims.begin(), h_dims.end());
    } else {
      H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, -1, "bad chunk data space");
    }

    if(native_type < 0) {
         H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_ARGS, H5E_BADTYPE, -1, "not a datatype");
    } else {
      opts.dtype = [&native_type]{
        if(H5Tequal(native_type, H5T_NATIVE_INT8) > 0) return pressio_int8_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_INT16) > 0) return pressio_int16_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_INT32) > 0) return pressio_int32_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_INT64) > 0) return pressio_int64_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_UINT8) > 0) return pressio_uint8_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_UINT16) > 0) return pressio_uint16_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_UINT32) > 0) return pressio_uint32_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_UINT64) > 0) return pressio_uint64_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_FLOAT) > 0) return pressio_float_dtype;
        if(H5Tequal(native_type, H5T_NATIVE_DOUBLE) > 0) return pressio_double_dtype;
        return pressio_byte_dtype;
      }();
      H5Tclose(native_type);
    }
    //get the compressor id and configuration from the dcpl
    if(H5Pexist(dcpl_id, "libpressio_compressor_options") > 0) {
      if(H5Pget(dcpl_id, "libpressio_compressor_options", &p_opts) > 0){
        H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get libpressio controls");
      } else {
        opts.options = *p_opts;
        delete p_opts;
      }
    }
    if(H5Pexist(dcpl_id, "libpressio_compressor_id") > 0) {
      if(H5Pget(dcpl_id, "libpressio_compressor_id", &p_string) > 0){
        H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get libpressio compressor_id");
      } else {
        opts.compressor_id = *p_string;
        delete p_string;
      }
    }

    //convert the options to cd_values
    cd_vals = get_cd_values_from_options(opts);

    if(H5Pget_filter_by_id(dcpl_id, H5Z_FILTER_LIBPRESSIO, &flags, 0, nullptr, 0, nullptr, nullptr) < 0){
         H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_CANTGET, 0, "unable to get current LIBPRESSIO flags");
    }

    //overwrite the plugin with the new configuration
    if(H5Pmodify_filter(dcpl_id, H5Z_FILTER_LIBPRESSIO, flags, cd_vals.size(), cd_vals.data()) < 0) {
      H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADVALUE, 0, "failed to modify cd_values");
    }

    

done:
    return retval;
  }

  /**
   * checks if the input data format is supported by the compressor
   *
   * since we don't know what the underlying compressor fully supports
   * we can only check some basics things here
   *
   * \param[in] type_id  datatype identifier passed in to H5Dcreate. should not be modified 
   * \returns non-negative value on success and a negative value for an error
   */
  static htri_t H5Z_libpressio_can_apply(hid_t , hid_t type_id,
                                         hid_t )
  {
    static char const*_funcname_ = "H5Z_libpressio_can_apply";
    htri_t retval = 0;
    H5T_class_t dclass;
    size_t dsize;
    hid_t native_type_id;

    if((dclass = H5Tget_class(type_id)) == H5T_NO_CLASS) {
      H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, -1, "bad datatype class");
    }

    if((dsize = H5Tget_size(type_id)) == 0) {
      H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, -1, "bad datatype size");
    }

    if(dclass == H5T_FLOAT) {
      switch(dsize) {
        case 4:
        case 8:
          break;
        default:
          H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0, "only 32 and 64 bit floats supported");
      }
    } else if(dclass == H5T_INTEGER) {
      switch(dsize) {
        case 1:
        case 2:
        case 4:
        case 8:
          break;
        default:
          H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0, "only 8, 16, 32, and 64 bit integers supported");
      }
    }

    native_type_id = H5Tget_native_type(type_id, H5T_DIR_ASCEND);
    if (H5Tget_order(type_id) != H5Tget_order(native_type_id)) {
        H5Z_LIBPRESSIO_PUSH_AND_GOTO(H5E_PLINE, H5E_BADTYPE, 0,
            "endian targetting is not currently supported with libpressio");
    }
    H5Tclose(native_type_id);

    retval = 1;
done:
    return retval;
  }

  const H5Z_class_t H5Z_LIBPRESSIO[1] = { {
    H5Z_CLASS_T_VERS,                    /* class_t version*/
    (H5Z_filter_t)H5Z_FILTER_LIBPRESSIO, /*filter id number*/
    1,                                   /*has an encoder phase*/
    1,                                   /*has an decoder phase*/
    "libpressio",                        /*filter name for debugging*/
    H5Z_libpressio_can_apply,            /* can apply callback??*/
    H5Z_libpressio_set_local,            /* set local callback??*/
    H5Z_filter_libpressio                /*actual filter function*/
  } };

  H5PL_type_t H5PLget_plugin_type() { return H5PL_TYPE_FILTER; }

  const void* H5PLget_plugin_info() { return H5Z_LIBPRESSIO; }

}
