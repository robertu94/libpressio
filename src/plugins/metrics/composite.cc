#include <vector>
#include <memory>
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/compat/std_compat.h"

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

  void begin_set_options(struct pressio_options const& options) override {
    for (auto& plugin : plugins) {
      plugin->begin_set_options(options);
    }
  }

  void end_set_options(struct pressio_options const& options, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_set_options(options, rc);
    }
  }

  void begin_compress(const struct pressio_data * input, struct pressio_data const * output) override {
    for (auto& plugin : plugins) {
      plugin->begin_compress(input, output);
    }
  }

  void end_compress(struct pressio_data const* input, pressio_data const * output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_compress(input, output, rc);
    }
  }

  void begin_decompress(struct pressio_data const* input, pressio_data const* output) override {
    for (auto& plugin : plugins) {
      plugin->begin_decompress(input, output);
    }
  }

  void end_decompress(struct pressio_data const* input, pressio_data const* output, int rc) override {
    for (auto& plugin : plugins) {
      plugin->end_decompress(input, output, rc);
    }
  }

  struct pressio_options get_metrics_results() const override {
    struct pressio_options metrics_result;
    for (auto const& plugin : plugins) {
      pressio_options plugin_options = plugin->get_metrics_results();
      auto tmp = pressio_options_merge(&metrics_result, &plugin_options);
      metrics_result = std::move(*tmp);
      pressio_options_free(tmp);
    }
    set_composite_metrics(metrics_result);

    return metrics_result;
  }

  struct pressio_options get_metrics_options() const override {
    struct pressio_options metrics_options;
    for (auto const& plugin : plugins) {
      pressio_options plugin_options = plugin->get_metrics_options();
      auto tmp = pressio_options_merge(&metrics_options, &plugin_options);
      metrics_options = std::move(*tmp);
      pressio_options_free(tmp);
    }
    set_composite_metrics(metrics_options);

    return metrics_options;
  }

  int set_metrics_options(pressio_options const& options) override {
    int rc = 0;
    for (auto const& plugin : plugins) {
      rc |= plugin->set_metrics_options(options);
    }
    return rc;
  }

  private:
  void set_composite_metrics(struct pressio_options& opt) const {
    //compression_rate
    {
      unsigned int compression_time, uncompressed_size;
      if(opt.get("time:compress", &compression_time) == pressio_options_key_set &&
         opt.get("size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        opt.set("composite:compression_rate", static_cast<double>(uncompressed_size)/compression_time);
      } else {
        opt.set_type("composite:compression_rate", pressio_option_double_type);
      }
    }

    //decompression_rate
    {
      unsigned int decompression_time, uncompressed_size;
      if (opt.get( "time:decompress", &decompression_time) == pressio_options_key_set &&
          opt.get("size:uncompressed_size", &uncompressed_size) == pressio_options_key_set) {
        opt.set("composite:decompression_rate", static_cast<double>(uncompressed_size)/decompression_time);
      } else {
        opt.set_type("composite:decompression_rate", pressio_option_double_type);
      }
    }

  }

  std::vector<std::unique_ptr<libpressio_metrics_plugin>> plugins;
};

std::unique_ptr<libpressio_metrics_plugin> make_m_composite(std::vector<std::unique_ptr<libpressio_metrics_plugin>>&& plugins) {
  return compat::make_unique<composite_plugin>(std::move(plugins));
}
