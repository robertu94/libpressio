#include <vector>
#include <memory>
#include <random>
#include <numeric>
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"

namespace libpressio { namespace sampling { 

class sample_compressor_plugin: public libpressio_compressor_plugin {
  public:
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set(options, "sample:mode", mode);
      set(options, "sample:seed", seed);
      set(options, "sample:rate", rate);
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set(options,"pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
      set(options,"pressio:stability", "unstable");
      set(options, "sample:mode", std::vector<std::string>{"wr", "wor", "decimate"});
      return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", "A \"compressor\" which samples the data by row");
      set(options, "sample:mode", R"(what kind of sampling to apply
      +  wr -- with replacement
      +  wor -- without replacement
      +  decimate -- sample every kth entry
      )");
      set(options, "sample:seed", "the seed to use");
      set(options, "sample:rate", "the sampling rate to use");
      return options;
    }


    int set_options_impl(struct pressio_options const& options) override {
      get(options, "sample:mode", &mode);
      get(options, "sample:seed", &seed);
      get(options, "sample:rate", &rate);
      return 0;
    }

    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      std::vector<size_t> const& dims = input->dimensions();
      size_t sample_size;
      size_t take_rows = 0;
      const size_t total_rows = dims.back();
      if(mode == "wr" || mode == "wor") {
        sample_size = std::floor(rate * total_rows);
      } else if( mode =="decimate") {
        do {
          sample_size = std::ceil(static_cast<double>(total_rows)/static_cast<double>(++take_rows));
        } while(rate < static_cast<double>(sample_size) / static_cast<double>(total_rows));
      } else {
        return invalid_mode(mode);
      }

      std::vector<size_t> rows_to_sample;
      std::seed_seq seed_s{seed};
      std::minstd_rand dist{seed_s};

      if(mode == "wr") {
        rows_to_sample.resize(sample_size);
        std::uniform_int_distribution<size_t> gen(0, total_rows-1);
        auto rand = [&]{return gen(dist); };
        std::generate(std::begin(rows_to_sample), std::end(rows_to_sample), rand);
        std::sort(std::begin(rows_to_sample), std::end(rows_to_sample));
      } else if (mode == "wor") {
        rows_to_sample.resize(total_rows);
        std::iota(std::begin(rows_to_sample), std::end(rows_to_sample), 0);
        std::shuffle(std::begin(rows_to_sample), std::end(rows_to_sample), dist);
        rows_to_sample.resize(sample_size);
        std::sort(std::begin(rows_to_sample), std::end(rows_to_sample));
      } else if (mode == "decimate") {
        size_t i = 0;
        std::generate(std::begin(rows_to_sample), std::end(rows_to_sample), [=]() mutable { size_t ret = i; i += take_rows; return ret; });
      } else {
        return 1;
      }

      //actually sample the "rows"
      std::vector<size_t> new_dims = dims;
      new_dims.back() = sample_size;
      *output = pressio_data::owning(input->dtype(), new_dims);
      unsigned char* output_ptr = static_cast<unsigned char*>(output->data());
      unsigned char* input_ptr = static_cast<unsigned char*>(input->data());
      size_t row_size = std::accumulate(
          std::next(compat::rbegin(dims)),
          compat::rend(dims),
          1ul, compat::multiplies<>{}
          ) * pressio_dtype_size(input->dtype());

      for (auto row : rows_to_sample) {
        memcpy(output_ptr, input_ptr + (row*row_size), row_size);
        output_ptr += row_size;
      }

      return 0;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      *output = pressio_data::clone(*input);
      return 0;
    }

    int major_version() const override {
      return 0;
    }
    int minor_version() const override {
      return 0;
    }
    int patch_version() const override {
      return 1;
    }

    const char* version() const override {
      return "0.0.1";
    }

    const char* prefix() const override {
      return "sample";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<sample_compressor_plugin>(*this);
    }


  private:
    std::string mode;
    int seed = 0;
    double rate;
    int invalid_mode(std::string const& mode) {
      return set_error(1, mode + " invalid mode");
    }
};

static pressio_register compressor_sampling_plugin(compressor_plugins(), "sample", [](){ return compat::make_unique<sample_compressor_plugin>(); });


} }
