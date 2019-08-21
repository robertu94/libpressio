#include <optional>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"

class size_plugin : public libpressio_metrics_plugin {
  public:

    void end_compress(struct pressio_data const* input, pressio_data const* output, int) override {
      uncompressed_size = pressio_data_get_bytes(input);
      compressed_size = pressio_data_get_bytes(output);
      compression_ratio =  static_cast<double>(*uncompressed_size)/ *compressed_size;
    }

    void end_decompress(struct pressio_data const* , pressio_data const* output, int) override {
      decompressed_size = pressio_data_get_bytes(output);
    }

  struct pressio_options* get_metrics_results() const override {
    pressio_options* opt = pressio_options_new();

    {
      const char* key = "size:compression_ratio";
      if(compression_ratio) {
        pressio_options_set_double(opt, key, *compression_ratio);
      } else {
        pressio_options_set_type(opt, key, pressio_option_double_type);
      }
    }

    auto set_or_size_t = [&opt](const char* key, std::optional<size_t> size) {
      if(size) {
        pressio_options_set_uinteger(opt, key, *size);
      } else {
        pressio_options_set_type(opt, key, pressio_option_uint32_type);
      }
    };

    set_or_size_t("size:compressed_size", compressed_size);
    set_or_size_t("size:uncompressed_size", uncompressed_size);
    set_or_size_t("size:decompressed_size", decompressed_size);

    return opt;
  }

  private:
    std::optional<double> compression_ratio;
    std::optional<size_t> uncompressed_size;
    std::optional<size_t> compressed_size;
    std::optional<size_t> decompressed_size;

};

std::unique_ptr<libpressio_metrics_plugin> make_m_size() {
  return std::make_unique<size_plugin>();
}
