#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"

class size_plugin : public libpressio_metrics_plugin {
  public:
    int end_compress_impl(struct pressio_data const* input, pressio_data const* output, int) override {
      if(!output) return set_error(1, "missing output");
      uncompressed_size = pressio_data_get_bytes(input);
      compressed_size = pressio_data_get_bytes(output);
      compression_ratio =  static_cast<double>(*uncompressed_size)/ static_cast<double>(*compressed_size);
      bit_rate = static_cast<double>(*compressed_size * 8 /*bits_per_byte*/)/static_cast<double>(pressio_data_num_elements(input));
      return 0;
    }

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int) override {
      decompressed_size = pressio_data_get_bytes(output);
      return 0;
    }

    int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int ) override {
   
      if(outputs.empty()) return set_error(1, "missing outputs");
      uncompressed_size = 0;
      compressed_size = 0;
      size_t num_elements = 0;
      for (auto const& input : inputs) {
        *uncompressed_size += pressio_data_get_bytes(input);
        num_elements += pressio_data_num_elements(input);
      }
      for (auto const& output : outputs) {
        *compressed_size += pressio_data_get_bytes(output);
      }

      compression_ratio =  static_cast<double>(*uncompressed_size)/ static_cast<double>(*compressed_size);
      bit_rate = static_cast<double>(*compressed_size * 8 /*bits_per_byte*/)/static_cast<double>(num_elements);
      return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& ,
                                   compat::span<const pressio_data* const> const& outputs, int ) override {
    decompressed_size = 0;
    for (auto const& output : outputs) {
      *decompressed_size += pressio_data_get_bytes(output);
    }
    return 0;
  }

  
  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "captures the size of compressed buffers");
    set(opt, "size:compressed_size", "compressed size in bytes");
    set(opt, "size:uncompressed_size", "uncompressed size in bytes");
    set(opt, "size:decompressed_size", "decompressed_size in bytes");
    set(opt, "size:compression_ratio", "uncompressed_size/compressed_size");
    set(opt, "size:bit_rate", bit_rate);
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) const override {
    pressio_options opt;

    auto set_or_double = [&opt, this](const char* key, compat::optional<double> size) {
      if(size) {
        set(opt, key, *size);
      } else {
        set_type(opt, key, pressio_option_double_type);
      }
    };

    auto set_or_size_t = [&opt, this](const char* key, compat::optional<size_t> size) {
      if(size) {
        set(opt, key, static_cast<unsigned int>(*size));
      } else {
        set_type(opt, key, pressio_option_uint32_type);
      }
    };

    set_or_size_t("size:compressed_size", compressed_size);
    set_or_size_t("size:uncompressed_size", uncompressed_size);
    set_or_size_t("size:decompressed_size", decompressed_size);
    set_or_double("size:compression_ratio", compression_ratio);
    set_or_double("size:bit_rate", bit_rate);

    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<size_plugin>(*this);
  }
  const char* prefix() const override {
    return "size";
  }

  private:
    compat::optional<double> compression_ratio;
    compat::optional<double> bit_rate;
    compat::optional<size_t> uncompressed_size;
    compat::optional<size_t> compressed_size;
    compat::optional<size_t> decompressed_size;

};

static pressio_register metrics_size_plugin(metrics_plugins(), "size", [](){ return compat::make_unique<size_plugin>(); });
