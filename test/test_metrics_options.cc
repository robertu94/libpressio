#include <gtest/gtest.h>
#include <iterator>
#include <map>
#include <sz.h>

#include "libpressio.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"
#include "make_input_data.h"

class hasoptoins_metric : public libpressio_metrics_plugin
{
public:
  
  int set_options(pressio_options const& options) override {
    options.get("hasoptions:value", &value);
    return 0;
  }

  pressio_options get_options() const override {
    pressio_options options;
    options.set("hasoptions:value", value);
    return options;
  }

  pressio_options get_metrics_results() const override {
    return {};
  }
  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<hasoptoins_metric>(*this);
  }

  const char* prefix() const override {
    return "hasoptions";
  }

  int value = 3;
};

//register the plugin in the under the names counts
static pressio_register X(metrics_plugins(), "hasoptions", []() {
  return compat::make_unique<hasoptoins_metric>();
});

TEST(ExternalPlugin, TestMetricHasOptions) {
  pressio library;

  //test the new one and the old ones
  const char* metrics_ids[] = {"hasoptions"};
  pressio_metrics metrics = library.get_metrics(std::begin(metrics_ids), std::end(metrics_ids));


  pressio_compressor compressor = library.get_compressor("sz");
  compressor->set_metrics(metrics);
  {
    auto metric_options = compressor->get_metrics_options();
    int value = 0;
    metric_options.get("hasoptions:value", &value);
    EXPECT_EQ(value, 3);
    metric_options.set("hasoptions:value", 4);
    compressor->set_metrics_options(metric_options);
  }

  {
    auto metric_options = compressor->get_metrics_options();
    int value = 0;
    metric_options.get("hasoptions:value", &value);
    EXPECT_EQ(value, 4);
  }

  {
    pressio_options* metric_options = pressio_compressor_metrics_get_options(&compressor);
    int value = 0;
    pressio_options_get_integer(metric_options, "hasoptions:value", &value);
    EXPECT_EQ(value, 4);
    pressio_options_set_integer(metric_options, "hasoptions:value", 5);
    pressio_compressor_metrics_set_options(&compressor, metric_options);
    pressio_options_free(metric_options);
  }

  {
    pressio_options* metric_options = pressio_compressor_metrics_get_options(&compressor);
    int value = 0;
    pressio_options_get_integer(metric_options, "hasoptions:value", &value);
    EXPECT_EQ(value, 5);
    pressio_options_free(metric_options);
  }

}
