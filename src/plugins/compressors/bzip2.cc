#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include <bzlib.h>
#include <cmath>

namespace libpressio { namespace compressors { namespace bzip2_ns {

class bzip2_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set(options, "bzip2:verbosity", verbosity);
    set(options, "bzip2:work_factor", workFactor);
    set(options, "pressio:lossless", blockSize100k);
    set(options, "bzip2:block_size_100k", blockSize100k);
    set(options, "bzip2:small", small);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    set(options, "pressio:lossless:min", 0);
    set(options, "pressio:lossless:max", 250);
    set(options, "bzip2:block_size_100k:min", 1);
    set(options, "bzip2:block_size_100k:max", 9);
    set(options, "bzip2:work_factor:min", 0);
    set(options, "bzip2:work_factor:max", 250);
    
        std::vector<std::string> invalidations {"bzip2:verbosity", "bzip2:small", "pressio:lossless", "bzip2:work_factor", "bzip2:block_size_100k"}; 
        std::vector<pressio_configurable const*> invalidation_children {}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, {}));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:lossless"}));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(the bzip2 lossless compressor https://sourceware.org/bzip2/)");
    set(options, "bzip2:work_factor", "between 0 and 250. How aggressively to try and compress data");
    set(options, "bzip2:block_size_100k", "between 1 and 9, what size block to consider at a time");
    set(options, "bzip2:verbosity", "use verbose logging to stdout if >0");
    set(options, "bzip2:small", "use a alternative decompression algorithm that uses less memory");
    set(options, "bzip2:block_size_100k:min", "min block_size_100k");
    set(options, "bzip2:block_size_100k:max", "max block_size_100k");
    set(options, "bzip2:work_factor:min", "min work_factor");
    set(options, "bzip2:work_factor:max", "max work_factor");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "bzip2:verbosity", &verbosity);
    get(options, "bzip2:small", &small);
    int temp = 0;
    if(get(options, "pressio:lossless", &temp) == pressio_options_key_set) {
      if(temp >= 0 && temp <= 250) {
        blockSize100k = temp;
      } else {
        set_error(1, "lossless out of range");
      }
    }
    if(get(options, "bzip2:work_factor", &temp) == pressio_options_key_set) {
      if(temp >= 0 && temp <= 250) {
        workFactor = temp;
      } else {
        set_error(1, "workFactor out of range");
      }
    }
    if( get(options, "bzip2:block_size_100k", &temp) == pressio_options_key_set) {
      if(temp >= 1 && temp <= 9) {
        blockSize100k = temp;
      } else {
        set_error(1, "blockSize100k out of range");
      }
    }
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* real_output) override
  {
    
    pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    pressio_data output =
        (real_output->has_data())
            ? (domain_manager().make_writeable(domain_plugins().build("malloc"),
                                               std::move(*real_output)))
            : (pressio_data::owning(pressio_byte_dtype,
                                    {static_cast<size_t>(std::ceil(
                                         static_cast<double>(input.size_in_bytes()) * 1.01)) +
                                     600}));
    if(input.size_in_bytes() > std::numeric_limits<unsigned int>::max()) {
      return set_error(1, "input is too large for bzip, max size is " + std::to_string(std::numeric_limits<unsigned int>::max()));
    }
    if( output.size_in_bytes() > std::numeric_limits<unsigned int>::max()) {
      return set_error(1, "output is too large for bzip, max size is " + std::to_string(std::numeric_limits<unsigned int>::max()));
    }
    unsigned int destLen = output.size_in_bytes();
    int rc = BZ2_bzBuffToBuffCompress(
        static_cast<char*>(output.data()),
        &destLen,
        static_cast<char*>(input.data()),
        input.size_in_bytes(),
        blockSize100k,
        verbosity,
        workFactor
        );
    output.set_dimensions({destLen});
    *real_output = std::move(output);

    return lp_bz_error(rc);
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* real_output) override
  {
    if(real_input->size_in_bytes() > std::numeric_limits<unsigned int>::max()) {
      return set_error(1, "input is too large for bzip, max size is " + std::to_string(std::numeric_limits<unsigned int>::max()));
    }
    if( real_output->size_in_bytes() > std::numeric_limits<unsigned int>::max()) {
      return set_error(1, "output is too large for bzip, max size is " + std::to_string(std::numeric_limits<unsigned int>::max()));
    }
    pressio_data input =
        domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    pressio_data output =
        (real_output->has_data())
            ? (domain_manager().make_writeable(domain_plugins().build("malloc"),
                                               std::move(*real_output)))
            : (pressio_data::owning(real_output->dtype(), real_output->dimensions()));
    unsigned int destLen = output.size_in_bytes();
    int rc = BZ2_bzBuffToBuffDecompress(
        static_cast<char*>(output.data()),
        &destLen,
        static_cast<char*>(input.data()),
        input.size_in_bytes(),
        small,
        verbosity);
    *real_output = std::move(output);

    return lp_bz_error(rc);
  }

  static std::array<int,3> parse_bz2version() {
    std::array<int,3> versions = {0,0,0};
    std::string version = BZ2_bzlibVersion();
    auto major_ver_div = version.find('.');
    auto minor_ver_div = version.find('.', major_ver_div + 1);
    auto patch_ver_div = version.find(',', minor_ver_div + 1);
    try {
      versions[0] = std::stoi(version.substr(0, major_ver_div));
      versions[1] = std::stoi(version.substr(major_ver_div+1, minor_ver_div-major_ver_div));
      versions[2] = std::stoi(version.substr(minor_ver_div+1, patch_ver_div-minor_ver_div));
    } catch(std::exception const&) {
      //pass
    }

    return versions;
  }

  int major_version() const override { return parse_bz2version().at(0); }
  int minor_version() const override { return parse_bz2version().at(1); }
  int patch_version() const override { return parse_bz2version().at(2); }
  const char* version() const override { return BZ2_bzlibVersion(); }
  const char* prefix() const override { return "bzip2"; }

  pressio_options get_metrics_results_impl() const override {
    return {};
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<bzip2_compressor_plugin>(*this);
  }

  int lp_bz_error(int rc) {
    switch(rc) {
      case BZ_RUN_OK:
      case BZ_FLUSH_OK:
      case BZ_FINISH_OK:
      case BZ_STREAM_END:
      case BZ_OK:
        return set_error(0, "");
      case BZ_CONFIG_ERROR:
        return set_error(1, "config error");
      case BZ_SEQUENCE_ERROR:
        return set_error(1, "sequence error");
      case BZ_PARAM_ERROR:
        return set_error(1, "param error");
      case BZ_MEM_ERROR:
        return set_error(1, "memory error");
      case BZ_DATA_ERROR:
        return set_error(1, "data error");
      case BZ_DATA_ERROR_MAGIC:
        return set_error(1, "data error magic");
      case BZ_IO_ERROR:
        return set_error(1, "io error");
      case BZ_UNEXPECTED_EOF:
        return set_error(1, "unexpected eof");
      case BZ_OUTBUFF_FULL:
        return set_error(1, "outbuf full");
      default:
        return set_error(2, "unknown error");
    }
  }

  int blockSize100k = 9;
  int workFactor = 30;
  int verbosity = 0;
  int small = 0;
};

pressio_register registration(compressor_plugins(), "bzip2", []() {
  return compat::make_unique<bzip2_compressor_plugin>();
});

} }}
