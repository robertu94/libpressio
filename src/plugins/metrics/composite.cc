#include <vector>
#include <memory>
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"

class composite_plugin : public libpressio_metrics_plugin {
  public:
  composite_plugin(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins) :
    plugins(std::move(plugins))
    {}
  void begin_check_options(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->begin_check_options(options);
    }
  }

  void end_check_options(struct pressio_options const* options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_check_options(options, rc);
    }
  }

  void begin_get_options() override {
    for (auto& plugin : plugins) {
      plugin->begin_get_options();
    }
  }

  void end_get_options(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->end_get_options(options);
    }
  }

  void begin_set_options(struct pressio_options const* options) override {
    for (auto& plugin : plugins) {
      plugin->begin_set_options(options);
    }
  }

  void end_set_options(struct pressio_options const* options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_set_options(options, rc);
    }
  }

  void begin_compress(const struct pressio_data * input, struct pressio_data * const * output) override {
    for (auto& plugin : plugins) {
      plugin->begin_compress(input, output);
    }
  }

  void end_compress(struct pressio_data const* input, pressio_data * const * output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_compress(input, output, rc);
    }
  }

  void begin_decompress(struct pressio_data const* input, pressio_data *const* output) override {
    for (auto& plugin : plugins) {
      plugin->begin_decompress(input, output);
    }
  }

  void end_decompress(struct pressio_data const* input, pressio_data *const* output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_decompress(input, output, rc);
    }
  }

  struct pressio_options* get_metrics_results() const override {
    struct pressio_options* opt = pressio_options_new();
    for (auto const& plugin : plugins) {
      auto plugin_options = plugin->get_metrics_results();
      auto tmp = pressio_options_merge(opt, plugin_options);
      pressio_options_free(opt);
      opt = tmp;
    }
    set_composite_metrics(opt);

    return opt;
  }

  private:
  void set_composite_metrics(struct pressio_options* opt) const {
    //compression_rate
    {
      unsigned int compression_time, uncompressed_size;
      if(pressio_options_get_uinteger(opt, "timer:compress", &compression_time) == pressio_options_key_set &&
         pressio_options_get_uinteger(opt, "size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        pressio_options_set_double(opt, "composite:compression_rate", static_cast<double>(uncompressed_size)/compression_time);
      }
    }

    //decompression_rate
    {
      unsigned int decompression_time, compressed_size;
      if(pressio_options_get_uinteger(opt, "timer:compress", &decompression_time) == pressio_options_key_set &&
         pressio_options_get_uinteger(opt, "size:uncompressed_size", &compressed_size) == pressio_options_key_set) {
        pressio_options_set_double(opt, "composite:decompression_rate", static_cast<double>(compressed_size)/decompression_time);
      }
    }

  }

  std::vector<std::unique_ptr<libpressio_metrics_plugin>> plugins;
};

std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins) {
  return std::make_unique<composite_plugin>(std::move(plugins));
}
