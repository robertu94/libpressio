#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include "cleanup.h"
#include <netcdf.h>

namespace libpressio { namespace netcdf_io_ns {

class netcdf_plugin : public libpressio_io_plugin {

  void set_error_nc(int err) {
    if(err != NC_NOERR) {
      throw std::runtime_error(nc_strerror(err));
    }
  }
  pressio_dtype from_nc_to_lp_dtype(nc_type type) {
    switch(type) {
      case NC_BYTE:
        return pressio_int8_dtype;
      case NC_SHORT:
        return pressio_int16_dtype;
      case NC_INT:
        return pressio_int32_dtype;
      case NC_INT64:
        return pressio_int64_dtype;
      case NC_UBYTE:
        return pressio_uint8_dtype;
      case NC_USHORT:
        return pressio_uint16_dtype;
      case NC_UINT:
        return pressio_uint32_dtype;
      case NC_UINT64:
        return pressio_uint64_dtype;
      case NC_FLOAT:
        return pressio_float_dtype;
      case NC_DOUBLE:
        return pressio_double_dtype;
      default:
        throw std::runtime_error("unknown ncdf_type");
    }
  }

  int read_nc(int file_id, int query_varid, pressio_data* data) {
    static_assert(sizeof(unsigned long long int) == 8, "netcdf assumes unsigned long long for 64 bit types");
    static_assert(sizeof(long long int) == 8, "netcdf assumes long long int for 64 bit types");

    void* data_ptr = data->data();
    switch(data->dtype()) {
      case pressio_int8_dtype:
        return nc_get_var_schar(file_id, query_varid, static_cast<int8_t*>(data_ptr));
      case pressio_int16_dtype:
        return nc_get_var_short(file_id, query_varid, static_cast<int16_t*>(data_ptr));
      case pressio_int32_dtype:
        return nc_get_var_int(file_id, query_varid, static_cast<int32_t*>(data_ptr));
      case pressio_int64_dtype:
        return nc_get_var_longlong(file_id, query_varid, static_cast<long long int*>(data_ptr));
      case pressio_byte_dtype:
        // intentional fall-though
      case pressio_uint8_dtype:
       return nc_get_var_uchar(file_id, query_varid, static_cast<uint8_t*>(data_ptr));
      case pressio_uint16_dtype:
       return nc_get_var_ushort(file_id, query_varid, static_cast<uint16_t*>(data_ptr));
      case pressio_uint32_dtype:
        return nc_get_var_uint(file_id, query_varid, static_cast<uint32_t*>(data_ptr));
      case pressio_uint64_dtype:
        return nc_get_var_ulonglong(file_id, query_varid, static_cast<unsigned long long int*>(data_ptr));
      case pressio_float_dtype:
        return nc_get_var_float(file_id, query_varid, static_cast<float*>(data_ptr));
      case pressio_double_dtype:
        return nc_get_var_double(file_id, query_varid, static_cast<double*>(data_ptr));
    }
    throw std::runtime_error("unexpected dtype");
  }

  public:
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
    try {
      int file_id;
      set_error_nc(nc_open(filename.c_str(), NC_NOWRITE, &file_id));
      auto cleanup_file = make_cleanup([&]{ nc_close(file_id);});

      int query_varid, ndims; 
      set_error_nc(nc_inq_varid(file_id, variable.c_str(), &query_varid));
      set_error_nc(nc_inq_varndims(file_id, query_varid, &ndims));

      nc_type type;
      set_error_nc(nc_inq_vartype(file_id, query_varid, &type));

      std::vector<int> dimids(ndims);
      set_error_nc(nc_inq_vardimid(file_id, query_varid, dimids.data()));

      size_t total_len = 1;
      std::vector<size_t> dimlen(ndims);
      for (int i = 0; i < ndims; ++i) {
        set_error_nc(nc_inq_dimlen(file_id, dimids[i], &dimlen[i]));
        total_len *= dimlen[i];
      }
      std::reverse(dimlen.begin(), dimlen.end());
      
      pressio_dtype lp_type = from_nc_to_lp_dtype(type);
      size_t size_in_bytes = total_len * pressio_dtype_size(lp_type);

      if(buf && buf->size_in_bytes() == size_in_bytes && buf->dtype() == lp_type) {
        set_error_nc(read_nc(file_id, query_varid, buf));
        return buf;
      } else {
        pressio_data* data = pressio_data_new_owning(
            lp_type,
            dimlen.size(),
            dimlen.data()
            );
        int err;
        if((err = read_nc(file_id, query_varid, data)) != NC_NOERR) {
          pressio_data_free(data);
          set_error_nc(err);
        }

        return data;
      }
    } catch(std::runtime_error const&ex) {
      set_error(2, ex.what());
      return nullptr;
    }
  }

  virtual int write_impl(struct pressio_data const*) override{
    return set_error(1, "writing not supported for netcdf at this time");
  }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "io:path", filename);
    set(options, "netcdf:variable", variable);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "io:path", &filename);
    get(options, "netcdf:variable", &variable);
    return 0;
  }

  const char* version() const override { return nc_inq_libvers(); }
  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "read netcdf files");
    set(opt, "io:path", "path to read files from");
    set(opt, "netcdf:variable", "variable to read data from");
    return opt;
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<netcdf_plugin>(*this);
  }
  const char* prefix() const override {
    return "netcdf";
  }

  private:
    std::string variable;
    std::string filename;
};

static pressio_register io_netcdf_plugin(io_plugins(), "netcdf", [](){ return compat::make_unique<netcdf_plugin>(); });
}}
