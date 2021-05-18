#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <sz.h>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"
#include "make_input_data.h"

class counting_metric: public libpressio_metrics_plugin {
  public:
  counting_metric() {
    //operator[] is non-const, so explicits instantiate each of the values we need
    counts[pressio_int8_dtype] = 0;
    counts[pressio_int16_dtype] = 0;
    counts[pressio_int32_dtype] = 0;
    counts[pressio_int64_dtype] = 0;
    counts[pressio_uint8_dtype] = 0;
    counts[pressio_uint16_dtype] = 0;
    counts[pressio_uint32_dtype] = 0;
    counts[pressio_uint64_dtype] = 0;
    counts[pressio_float_dtype] = 0;
    counts[pressio_double_dtype] = 0;
    counts[pressio_byte_dtype] = 0;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<counting_metric>(*this);
  }

  const char* prefix() const override {
    return "mycounts";
  }

  private:
  int begin_compress_impl(pressio_data const* input, pressio_data const*) override {
    counts[input->dtype()]++;
    return 0;
  }

  pressio_options get_documentation_impl() const override {
    pressio_options opts;
    opts.set("mycounts:int8", "number of data buffers of type int8");
    opts.set("mycounts:int16", "number of data buffers of type int16");
    opts.set("mycounts:int32", "number of data buffers of type int32");
    opts.set("mycounts:int64", "number of data buffers of type int64");
    opts.set("mycounts:uint8", "number of data buffers of type uint8");
    opts.set("mycounts:uint16", "number of data buffers of type uint16");
    opts.set("mycounts:uint32", "number of data buffers of type uint32");
    opts.set("mycounts:uint64", "number of data buffers of type uint64");
    opts.set("mycounts:float", "number of data buffers of type float");
    opts.set("mycounts:double", "number of data buffers of type double");
    opts.set("mycounts:byte", "number of data buffers of type byte");
    return opts;
  }

  pressio_options get_metrics_results(pressio_options const &options) const override {
    pressio_options opts;
    opts.set("mycounts:int8", counts.at(pressio_int8_dtype));
    opts.set("mycounts:int16", counts.at(pressio_int16_dtype));
    opts.set("mycounts:int32", counts.at(pressio_int32_dtype));
    opts.set("mycounts:int64", counts.at(pressio_int64_dtype));
    opts.set("mycounts:uint8", counts.at(pressio_uint8_dtype));
    opts.set("mycounts:uint16", counts.at(pressio_uint16_dtype));
    opts.set("mycounts:uint32", counts.at(pressio_uint32_dtype));
    opts.set("mycounts:uint64", counts.at(pressio_uint64_dtype));
    opts.set("mycounts:float", counts.at(pressio_float_dtype));
    opts.set("mycounts:double", counts.at(pressio_double_dtype));
    opts.set("mycounts:byte", counts.at(pressio_byte_dtype));
    return opts;
  }

  std::map<pressio_dtype, unsigned int> counts;
};

//register the plugin in the under the names counts
static pressio_register X(metrics_plugins(), "mycounts", [](){ return compat::make_unique<counting_metric>(); });

TEST(ExternalPlugin, TestMetricCounts) {
  pressio library;

  //test the new one and the old ones
  const char* metrics_ids[] = {"mycounts", "size"};
  pressio_metrics metrics = library.get_metrics(std::begin(metrics_ids), std::end(metrics_ids));


  auto compressor = library.get_compressor("sz");
  compressor->set_metrics(metrics);
  auto options = compressor->get_options();
  options.set("sz:error_bound_mode", ABS);
  options.set("sz:abs_err_bound", 0.5);


  if(compressor->check_options(options)) {
    std::cerr << compressor->error_msg() << std::endl;
    exit(compressor->error_code());
  }

  if(compressor->set_options(options)) {
    std::cerr << compressor->error_msg() << std::endl;
    exit(compressor->error_code());
  }

  double* rawinput_data = make_input_data();
  //providing a smaller than expected buffer to save time during testing
  std::vector<size_t> dims{30,30,30};

  auto input = pressio_data::move(pressio_double_dtype, rawinput_data, dims, pressio_data_libc_free_fn, nullptr);

  auto compressed = pressio_data::empty(pressio_byte_dtype, {});

  auto decompressed = pressio_data::empty(pressio_double_dtype, dims);

  if(compressor->compress(&input, &compressed)) {
    std::cerr << library.err_msg() << std::endl;
    exit(library.err_code());
  }

  if(compressor->decompress(&compressed, &decompressed)) {
    std::cerr << library.err_msg() << std::endl;
    exit(library.err_code());
  }

  auto metrics_results = compressor->get_metrics_results();

  unsigned int count;
  metrics_results.get("mycounts:double", &count);
  EXPECT_EQ(count, 1);

  metrics_results.get("mycounts:float", &count);
  EXPECT_EQ(count, 0);
}
