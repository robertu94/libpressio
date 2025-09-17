
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"
#include "cleanup.h"
#include <matio.h>

namespace libpressio { namespace io { namespace matio_ns {

    pressio_dtype to_pressio_dtype(matio_types type) {
        switch(type) {
            case MAT_T_INT8:
                return pressio_int8_dtype;
            case MAT_T_INT16:
                return pressio_int16_dtype;
            case MAT_T_INT32:
                return pressio_int32_dtype;
            case MAT_T_INT64:
                return pressio_int64_dtype;
            case MAT_T_UINT8:
                return pressio_uint8_dtype;
            case MAT_T_UINT16:
                return pressio_uint16_dtype;
            case MAT_T_UINT32:
                return pressio_uint32_dtype;
            case MAT_T_UINT64:
                return pressio_uint64_dtype;
            case MAT_T_SINGLE:
                return pressio_float_dtype;
            case MAT_T_DOUBLE:
                return pressio_double_dtype;
            default:
                throw std::runtime_error("matio unsupported type");
        }
    }

    matio_types to_matio_type(pressio_dtype type) {
        switch(type) {
             case pressio_int8_dtype: return MAT_T_INT8;
             case pressio_int16_dtype: return MAT_T_INT16;
             case pressio_int32_dtype: return MAT_T_INT32;
             case pressio_int64_dtype: return MAT_T_INT64;
             case pressio_uint8_dtype: return MAT_T_UINT8;
             case pressio_uint16_dtype: return MAT_T_UINT16;
             case pressio_uint32_dtype: return MAT_T_UINT32;
             case pressio_uint64_dtype: return MAT_T_UINT64;
             case pressio_float_dtype: return MAT_T_SINGLE;
             case pressio_double_dtype: return MAT_T_DOUBLE;
            default:
                throw std::runtime_error("pressio_dtype unsupported for matio");
        }
    }
    matio_classes to_matio_ctype(pressio_dtype type) {
        switch(type) {
             case pressio_int8_dtype: return MAT_C_INT8;
             case pressio_int16_dtype: return MAT_C_INT16;
             case pressio_int32_dtype: return MAT_C_INT32;
             case pressio_int64_dtype: return MAT_C_INT64;
             case pressio_uint8_dtype: return MAT_C_UINT8;
             case pressio_uint16_dtype: return MAT_C_UINT16;
             case pressio_uint32_dtype: return MAT_C_UINT32;
             case pressio_uint64_dtype: return MAT_C_UINT64;
             case pressio_float_dtype: return MAT_C_SINGLE;
             case pressio_double_dtype: return MAT_C_DOUBLE;
            default:
                throw std::runtime_error("pressio_dtype unsupported for matio");
        }
    }


class matio_plugin : public libpressio_io_plugin {
  public:
  virtual struct pressio_data* read_impl(struct pressio_data*) override {
      try {
        mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDONLY);
        auto close_mat = make_cleanup([mat]{ Mat_Close(mat);});

        matvar_t* matvar = Mat_VarRead(mat, varname.c_str());
        auto close_matvar = make_cleanup([matvar]{ Mat_VarFree(matvar);});

        auto dims = std::vector(matvar->dims, matvar->dims + matvar->rank);
        std::reverse(dims.begin(), dims.end());

        return new pressio_data(pressio_data::copy(
                to_pressio_dtype(matvar->data_type),
                matvar->data,
                dims
                ));

      } catch (std::runtime_error const& ex) {
        set_error(1, ex.what());
        return nullptr;
      }
    }
  virtual int write_impl(struct pressio_data const* indata) override{
      pressio_data data = domain_manager().make_readable(domain_plugins().build("malloc"),*indata);
      try {
        mat_t* mat = Mat_Open(filename.c_str(), MAT_ACC_RDWR);
        auto close_mat = make_cleanup([mat]{ Mat_Close(mat);});

        auto matio_dims = data.dimensions();
        std::reverse(matio_dims.begin(), matio_dims.end());
        matvar_t* matvar = Mat_VarCreate(varname.c_str(), to_matio_ctype(data.dtype()),  to_matio_type(data.dtype()), static_cast<int>(matio_dims.size()), matio_dims.data(), data.data(), 0);

        auto free_matvar = make_cleanup([matvar]{Mat_VarFree(matvar);});

        matvar_t* exists = Mat_VarReadInfo(mat, varname.c_str());
        if(exists != nullptr) {
            Mat_VarFree(exists);
            Mat_VarDelete(mat, varname.c_str());
        }

        if(Mat_VarWrite(mat, matvar, MAT_COMPRESSION_NONE) != 0) {
            throw std::runtime_error("matio writing failed");
        }

        return 0;
      } catch(std::runtime_error const& ex){
          return set_error(1, ex.what());
      }
    }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "io:path", filename);
    set(options, "matio:varname", varname);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "io:path", &filename);
    get(options, "matio:varname", &varname);
    return 0;
  }

  const char* version() const override { return "0.0.1"; }
  int major_version() const override { return MATIO_MAJOR_VERSION; }
  int minor_version() const override { return MATIO_MINOR_VERSION; }
  int patch_version() const override { return MATIO_RELEASE_LEVEL; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "read data from matlab matrix files");
    set(opt, "io:path", "path to the file");
    set(opt, "matio:varname", "variable in the file");
    return opt;
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<matio_plugin>(*this);
  }
  const char* prefix() const override {
    return "matio";
  }

  private:

  std::string filename;
  std::string varname;
};

pressio_register registration(io_plugins(), "matio", [](){ return compat::make_unique<matio_plugin>(); });
}}}
