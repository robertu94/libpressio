
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"
#include "stdio.h"
#include "eccodes.h"
#include "pressio_posix.h"
#include "cleanup.h"

namespace libpressio { namespace grib_io_ns {

class grib_plugin : public libpressio_io_plugin {
  public:
  virtual struct pressio_data* read_impl(struct pressio_data* buf) override {
        FILE* in = fopen(path.c_str(), "r");
        if(in == nullptr) {
            set_error(1, errno_to_error());
            return nullptr;
        }
        auto cleanup_file = make_cleanup([in]{ fclose(in);});

        codes_handle* codes = nullptr;
        int err = 0;
        std::map<long, pressio_data> levels;
        while((codes = codes_handle_new_from_file(nullptr, in, PRODUCT_GRIB, &err)) != NULL) {
            auto cleanup_handle = make_cleanup([&codes]{codes_handle_delete(codes);});
            size_t name_length = 0;
            if(codes_get_length(codes, "shortName", &name_length) != 0) {
                set_error(2, "failed to read shortName length");
                return nullptr;
            }
            std::string message_name (name_length, '\0');
            codes_get_string(codes, "shortName", message_name.data(), &name_length);
            message_name.resize(name_length - 1); //remove trailing null
            if (message_name != shortName) {
                continue;
            }

            long Ni = 0 , Nj = 0, level = 0;
            if(codes_get_long(codes, "Ni", &Ni) != 0) {
                set_error(2, "failed to read Ni");
                return nullptr;
            }
            if (codes_get_long(codes, "Nj", &Nj) != 0) {
                set_error(2, "failed to read Nj");
                return nullptr;
            }
            if (codes_get_long(codes, "level", &level) != 0) {
                set_error(2, "failed to read Nj");
                return nullptr;
            }

            //confirm the datatype
            int type;
            if(codes_get_native_type(codes, "values", &type) !=  0) {
                set_error(2, "failed to read type of values");
                return nullptr;
            }

            //confirm the size
            size_t real_size;
            if(codes_get_size(codes, "values", &real_size) !=  0) {
                set_error(2, "failed to read size of values");
                return nullptr;
            }
            if(real_size != static_cast<size_t>(Ni*Nj)) {
                set_error(2, "real size does not equal expected size, format not supported by libpressio");
                return nullptr;
            }


            switch(type) {
                case CODES_TYPE_DOUBLE:
                    {
                        levels.emplace(level, pressio_data::owning(pressio_double_dtype, {size_t(Ni),size_t(Nj)}));
                        codes_get_double_array(codes, "values", static_cast<double*>(levels[level].data()), &real_size);
                        break;
                    }
                default:
                    {
                    std::string err = "unexpected type in grib file ";
                    err += codes_get_type_name(type);
                    set_error(3, err);
                    return nullptr;
                    }
            }
        }

        if(levels.empty()) {
            set_error(6, "dataset not found " + shortName);
            return nullptr;
        }

        const std::vector<size_t> expected_dim_size = levels.begin()->second.dimensions();
        std::vector<size_t> output_dims = levels.begin()->second.dimensions();
        pressio_dtype expected_dtype = levels.begin()->second.dtype();
        output_dims.emplace_back(levels.size());

        pressio_data* out =
            new pressio_data((buf != nullptr)
                ? domain_manager().make_writeable(domain_plugins().build("malloc"), std::move(*buf))
                : pressio_data::owning(expected_dtype, output_dims));
        size_t offset = 0;
        for(auto const& it: levels) {
            //check dimensions are what are expected
            if(it.second.dimensions() != expected_dim_size || it.second.dtype() != expected_dtype) {
                set_error(5, "dimensions or types of all levels are not equal in GRIB");
                return nullptr;
            }
            memcpy(static_cast<uint8_t*>(out->data()) + offset, it.second.data(), it.second.size_in_bytes());
            offset += it.second.size_in_bytes();
        }

        return out;
    }
  virtual int write_impl(struct pressio_data const*) override{
        return set_error(1, "not supported");
    }

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "io:path", path);
    set(options, "grib:short_name", shortName);
    set(options, "grib:namespace", namespace_id);
    return options;
  }

  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "io:path", &path);
    get(options, "grib:short_name", &shortName);
    get(options, "grib:namespace", &namespace_id);
    return 0;
  }

  const char* version() const override { return "0.0.1"; }
  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "experimental");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "uses eccodes to read grib files");
    set(opt, "grib:short_name", "what short variable to access");
    set(opt, "grib:namespace", "what namespace to access, the empty string does not use a namespace");
    return opt;
  }

  std::shared_ptr<libpressio_io_plugin> clone() override {
    return compat::make_unique<grib_plugin>(*this);
  }
  const char* prefix() const override {
    return "grib";
  }

  private:
    std::string path;
    std::string shortName;
    std::string namespace_id;
};

static pressio_register io_grib_plugin(io_plugins(), "grib", [](){ return compat::make_unique<grib_plugin>(); });
}}

