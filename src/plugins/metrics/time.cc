#include <chrono>
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/std_compat.h"

using std::chrono::high_resolution_clock;
using std::chrono::time_point;
using std::chrono::duration_cast;
using std::chrono::milliseconds;

namespace {
  struct time_range{
    time_point<high_resolution_clock> begin;
    time_point<high_resolution_clock> end;
    unsigned int elapsed() const { return duration_cast<milliseconds>(end-begin).count(); }
  };
  using timer = compat::optional<time_range>;
}

class time_plugin : public libpressio_metrics_plugin {
  public:

  void begin_check_options(struct pressio_options const* ) override {
    check_options = time_range();
    check_options->begin = high_resolution_clock::now();
  }

  void end_check_options(struct pressio_options const*, int ) override {
    check_options->end = high_resolution_clock::now();
  }

  void begin_get_options() override {
    get_options = time_range();
    get_options->begin = high_resolution_clock::now();
  }

  void end_get_options(struct pressio_options const* ) override {
    get_options->end = high_resolution_clock::now();
  }

  void begin_get_configuration() override {
    get_configuration = time_range();
    get_configuration->begin = high_resolution_clock::now();
  }

  void end_get_configuration(struct pressio_options const* ) override {
    get_configuration->end = high_resolution_clock::now();
  }


  void begin_set_options(struct pressio_options const* ) override {
    set_options = time_range();
    set_options->begin = high_resolution_clock::now();
  }

  void end_set_options(struct pressio_options const* , int ) override {
    set_options->end = high_resolution_clock::now();
  }

  void begin_compress(const struct pressio_data * , struct pressio_data const * ) override {
    compress = time_range();
    compress->begin = high_resolution_clock::now();
  }

  void end_compress(struct pressio_data const* , pressio_data const * , int ) override {
    compress->end = high_resolution_clock::now();
  }

  void begin_decompress(struct pressio_data const* , pressio_data const* ) override {
    decompress = time_range();
    decompress->begin = high_resolution_clock::now();
  }

  void end_decompress(struct pressio_data const* , pressio_data const* , int ) override {
    decompress->end = high_resolution_clock::now();
  }

  struct pressio_options* get_metrics_results() const override {
    struct pressio_options* opt = pressio_options_new();

    auto set_or = [&opt](const char* key, timer time) {
      if(time) {
        pressio_options_set_uinteger(opt, key, time->elapsed());
      } else {
        pressio_options_set_type(opt, key, pressio_option_uint32_type);
      }
    };

    set_or("time:check_options", check_options);
    set_or("time:set_options", set_options);
    set_or("time:get_options", get_options);
    set_or("time:compress", compress);
    set_or("time:decompress", decompress);

    return opt;
  }

  private:
  timer check_options;
  timer set_options;
  timer get_options;
  timer get_configuration;
  timer compress;
  timer decompress;
};

static pressio_register X(metrics_plugins(), "time", [](){ return compat::make_unique<time_plugin>(); });
