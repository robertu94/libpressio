#include <gtest/gtest.h>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace {
const char* input_keys[] = {"dist_qoi:orig_data_key", "dist_qoi:dec_data_key"};

// Return fixed fixtures, independent of configuration.
class dist_qoi_mock_metric : public libpressio::metrics::libpressio_metrics_plugin {
public:
  pressio_options get_documentation_impl() const override { return {}; }
  pressio_options get_metrics_results(pressio_options const&) override {
    pressio_options results;
    set(results, "dist_qoi_mock:statistics_original", pressio_data{0.0, 2.0, 0.0, 2.0});
    set(results, "dist_qoi_mock:statistics_decompressed", pressio_data{4.0, 1.0, 3.0, 0.0});
    set(results, "dist_qoi_mock:original", pressio_data{0.0, 2.0});
    set(results, "dist_qoi_mock:reconstructed", pressio_data{1.0, 3.0});
    set(results, "dist_qoi_mock:nan", pressio_data{1.0, std::numeric_limits<double>::quiet_NaN()});
    set(results, "dist_qoi_mock:positive_infinity", pressio_data{1.0, std::numeric_limits<double>::infinity()});
    set(results, "dist_qoi_mock:negative_infinity", pressio_data{1.0, -std::numeric_limits<double>::infinity()});
    return results;
  }
  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<dist_qoi_mock_metric>(*this);
  }
  const char* prefix() const override { return "dist_qoi_mock"; }
};

libpressio::pressio_register registration(libpressio::metrics_plugins(), "dist_qoi_mock", []() {
  return compat::make_unique<dist_qoi_mock_metric>();
});

class DistQoi : public ::testing::Test {
protected:
  void SetUp() override {
    metric = libpressio::metrics_plugins().build("dist_qoi");
    if (!metric) GTEST_SKIP() << "build with the dist_qoi metrics plugin enabled";
  }

  void run(std::initializer_list<std::pair<const char*, const char*>> inputs = {}) {
    pressio_options options;
    options.set("dist_qoi:metric", "dist_qoi_mock");
    for (auto key : input_keys) options.set(key, "");
    for (auto const& input : inputs) options.set(input.first, input.second);
    ASSERT_EQ(metric->set_options(options), 0);
    results = metric->get_metrics_results({});
  }

  template<class T>
  T result(const char* key) {
    SCOPED_TRACE(key);
    T value{};
    EXPECT_EQ(results.get(key, &value), pressio_options_key_set);
    return value;
  }

  void expect_absent(std::initializer_list<const char*> keys) {
    for (auto key : keys) {
      EXPECT_FALSE(results.get(key).has_value()) << key;
    }
  }

  pressio_metrics metric;
  pressio_options results;
};

TEST_F(DistQoi, RejectsNonFiniteInputsAndRecovers) {
  for (auto input : input_keys) {
    for (auto key : {"dist_qoi_mock:nan", "dist_qoi_mock:positive_infinity", "dist_qoi_mock:negative_infinity"}) {
      SCOPED_TRACE(std::string(input) + ": " + key);
      ASSERT_NO_FATAL_FAILURE(run({{input, key}}));
      EXPECT_EQ(result<int32_t>("dist_qoi:error_code"), 1);
      auto message = result<std::string>("dist_qoi:error_msg");
      EXPECT_NE(message.find(key), std::string::npos);
      EXPECT_NE(message.find("index 1"), std::string::npos);
      EXPECT_TRUE(results.get("dist_qoi_mock:original").has_value());
      expect_absent({"dist_qoi:data", "dist_qoi:mean", "dist_qoi:wasserstein_distance"});
    }
  }
  ASSERT_NO_FATAL_FAILURE(run({{"dist_qoi:orig_data_key", "dist_qoi_mock:statistics_original"},
                               {"dist_qoi:dec_data_key", "dist_qoi_mock:statistics_decompressed"}}));
  EXPECT_EQ(result<int32_t>("dist_qoi:error_code"), 0);
  EXPECT_TRUE(result<std::string>("dist_qoi:error_msg").empty());
  EXPECT_TRUE(results.get("dist_qoi:data").has_value());
}

TEST_F(DistQoi, PublishesDistanceStatistics) {
  ASSERT_NO_FATAL_FAILURE(run({{"dist_qoi:orig_data_key", "dist_qoi_mock:statistics_original"},
                               {"dist_qoi:dec_data_key", "dist_qoi_mock:statistics_decompressed"}}));
  const std::pair<const char*, double> expected[] = {
      {"dist_qoi:mean", 2.5}, {"dist_qoi:min", 1.0}, {"dist_qoi:max", 4.0},
      {"dist_qoi:median", 2.5}, {"dist_qoi:p90", 3.7}, {"dist_qoi:p99", 3.97}, {"dist_qoi:p999", 3.997}};
  for (auto const& item : expected) {
    EXPECT_NEAR(result<double>(item.first), item.second, 1e-12) << item.first;
  }
  pressio_data raw;
  ASSERT_EQ(results.get("dist_qoi:data", &raw), pressio_options_key_set);
  const std::vector<double> expected_data = {4.0, 1.0, 3.0, 2.0};
  ASSERT_EQ(raw.num_elements(), expected_data.size());
  const double* ptr = static_cast<const double*>(raw.data());
  ASSERT_NE(ptr, nullptr);
  for (size_t i = 0; i < expected_data.size(); ++i) EXPECT_DOUBLE_EQ(ptr[i], expected_data[i]);
  EXPECT_TRUE(results.get("dist_qoi:wasserstein_distance").has_value());
}

TEST_F(DistQoi, PublishesWassersteinAndAbsoluteErrors) {
  ASSERT_NO_FATAL_FAILURE(run({{"dist_qoi:orig_data_key", "dist_qoi_mock:original"},
                              {"dist_qoi:dec_data_key", "dist_qoi_mock:reconstructed"}}));
  EXPECT_NEAR(result<double>("dist_qoi:wasserstein_distance"), 1.0, 1e-12);
  EXPECT_DOUBLE_EQ(result<double>("dist_qoi:mean"), 1.0);
  EXPECT_TRUE(results.get("dist_qoi:data").has_value());
}

TEST_F(DistQoi, RejectsUnequalSizes) {
  ASSERT_NO_FATAL_FAILURE(run({{"dist_qoi:orig_data_key", "dist_qoi_mock:original"},
                              {"dist_qoi:dec_data_key", "dist_qoi_mock:statistics_decompressed"}}));
  EXPECT_EQ(result<int32_t>("dist_qoi:error_code"), 3);
  EXPECT_NE(result<std::string>("dist_qoi:error_msg").find("same number of elements"), std::string::npos);
  expect_absent({"dist_qoi:mean", "dist_qoi:data", "dist_qoi:wasserstein_distance"});
}

TEST_F(DistQoi, EmptyKeysDisableLookupAndConfiguredKeysAreRequired) {
  for (auto key : input_keys) {
    std::string value;
    EXPECT_EQ(metric->get_options().get(key, &value), pressio_options_key_set);
    EXPECT_TRUE(value.empty()) << key;
  }
  ASSERT_NO_FATAL_FAILURE(run());
  EXPECT_EQ(result<int32_t>("dist_qoi:error_code"), 0);
  // Neither finite nor non-finite arrays are read implicitly.
  expect_absent({"dist_qoi:mean", "dist_qoi:data", "dist_qoi:wasserstein_distance"});

  for (auto input : input_keys) {
    SCOPED_TRACE(input);
    ASSERT_NO_FATAL_FAILURE(run({{input, "dist_qoi_mock:missing"}}));
    EXPECT_EQ(result<int32_t>("dist_qoi:error_code"), 2);
    EXPECT_NE(result<std::string>("dist_qoi:error_msg").find("dist_qoi_mock:missing"), std::string::npos);
  }
}
} // namespace
