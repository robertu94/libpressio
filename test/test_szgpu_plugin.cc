#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <thread>
#include <vector>
#include <szgpu/szgpu.h>
#include "gtest/gtest.h"
#include "libpressio_ext/cpp/libpressio.h"
#include "libpressio_ext/cpp/domain_manager.h"

using namespace libpressio;

namespace {
std::vector<size_t> const dims{100, 50, 37};

template <class T>
pressio_data make_field() {
  pressio_data data = pressio_data::owning(pressio_dtype_from_type<T>(), dims);
  T* ptr = static_cast<T*>(data.data());
  for (size_t k = 0; k < dims[2]; ++k)
    for (size_t j = 0; j < dims[1]; ++j)
      for (size_t i = 0; i < dims[0]; ++i)
        *ptr++ = static_cast<T>(10 * std::sin(0.05 * i) * std::cos(0.07 * j) + 0.1 * k);
  return data;
}

pressio_data to_host(pressio_data const& data) {
  return domain_manager().make_readable(domain_plugins().build("malloc"), data);
}

template <class T>
double max_abs_error(pressio_data const& original, pressio_data const& decompressed) {
  pressio_data host = to_host(decompressed);
  T const* expected = static_cast<T const*>(original.data());
  T const* actual = static_cast<T const*>(host.data());
  double max_error = 0;
  for (size_t i = 0; i < original.num_elements(); ++i) {
    double const error = std::fabs(static_cast<double>(expected[i]) - static_cast<double>(actual[i]));
    if (!std::isfinite(error)) return std::numeric_limits<double>::infinity();
    max_error = std::max(max_error, error);
  }
  return max_error;
}
}

class SZGPUPlugin : public ::testing::Test {
  protected:
    void SetUp() override {
      ASSERT_NE(compressor.plugin, nullptr) << library.err_msg();
      ASSERT_EQ(compressor->set_options({{"pressio:abs", abs_bound}}), 0) << compressor->error_msg();
    }

    template <class T>
    void host_round_trip() {
      pressio_data input = make_field<T>();
      pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
      pressio_data decompressed = pressio_data::owning(input.dtype(), input.dimensions());
      ASSERT_EQ(compressor->compress(&input, &compressed), 0) << compressor->error_msg();
      EXPECT_EQ(compressed.dtype(), pressio_byte_dtype);
      EXPECT_GT(compressed.size_in_bytes(), 0u);
      EXPECT_LT(compressed.size_in_bytes(), input.size_in_bytes());
      ASSERT_EQ(compressor->decompress(&compressed, &decompressed), 0) << compressor->error_msg();
      EXPECT_EQ(decompressed.dtype(), input.dtype());
      EXPECT_EQ(decompressed.dimensions(), input.dimensions());
      EXPECT_LE(max_abs_error<T>(input, decompressed), abs_bound * 1.01);
    }

    double const abs_bound = 1e-3;
    pressio library;
    pressio_compressor compressor = library.get_compressor("szgpu");
};

TEST_F(SZGPUPlugin, ReportsSZGPUVersion) {
  EXPECT_EQ(compressor->major_version(), SZGPU_VERSION_MAJOR);
  EXPECT_EQ(compressor->minor_version(), SZGPU_VERSION_MINOR);
  EXPECT_EQ(compressor->patch_version(), SZGPU_VERSION_PATCH);
  EXPECT_STREQ(compressor->version(), SZGPU_VERSION_STRING);
}

TEST_F(SZGPUPlugin, HostRoundTripFloat) { host_round_trip<float>(); }
TEST_F(SZGPUPlugin, HostRoundTripDouble) { host_round_trip<double>(); }

TEST_F(SZGPUPlugin, DeviceBuffersAreUsedInPlace) {
  pressio_data host_input = make_field<float>();
  pressio_data no_data = pressio_data::empty(pressio_float_dtype, dims);
  pressio_data estimate = pressio_data::empty(pressio_byte_dtype, {});
  ASSERT_EQ(compressor->compress(&no_data, &estimate), 0) << compressor->error_msg();
  EXPECT_EQ(estimate.size_in_bytes(), szgpu_max_compressed_bytes(host_input.num_elements()));

  auto cuda = domain_plugins().build("cudamalloc");
  pressio_data input = domain_manager().make_readable(cuda, host_input);
  pressio_data compressed = pressio_data::owning(pressio_byte_dtype, {estimate.size_in_bytes()}, cuda);
  pressio_data decompressed = pressio_data::owning(pressio_float_dtype, dims, cuda);
  void* const compressed_ptr = compressed.data();
  void* const decompressed_ptr = decompressed.data();

  ASSERT_EQ(compressor->compress(&input, &compressed), 0) << compressor->error_msg();
  EXPECT_EQ(compressed.data(), compressed_ptr);
  EXPECT_LT(compressed.size_in_bytes(), estimate.size_in_bytes());
  ASSERT_EQ(compressor->decompress(&compressed, &decompressed), 0) << compressor->error_msg();
  EXPECT_EQ(decompressed.data(), decompressed_ptr);
  EXPECT_LE(max_abs_error<float>(host_input, decompressed), abs_bound * 1.01);
}

TEST_F(SZGPUPlugin, StreamMatchesNativeSZGPU) {
  pressio_data input = make_field<double>();
  pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
  ASSERT_EQ(compressor->compress(&input, &compressed), 0) << compressor->error_msg();
  pressio_data host_compressed = to_host(compressed);

  size_t const n = input.num_elements();
  std::vector<unsigned char> native(szgpu_max_compressed_bytes(n));
  size_t native_size = 0;
  ASSERT_EQ(szgpu_compress_hostptr_f64(static_cast<const double*>(input.data()), native.data(), n, abs_bound, &native_size), SZGPU_STATUS_OK);
  ASSERT_EQ(host_compressed.size_in_bytes(), native_size);
  EXPECT_EQ(std::memcmp(host_compressed.data(), native.data(), native_size), 0);
}

TEST_F(SZGPUPlugin, RelativeBoundThroughPressioMetaCompressor) {
  double const rel_bound = 1e-4;
  pressio_compressor rel = library.get_compressor("pressio");
  ASSERT_NE(rel.plugin, nullptr) << library.err_msg();
  ASSERT_EQ(rel->set_options({{"pressio:compressor", std::string("szgpu")}}), 0) << rel->error_msg();
  ASSERT_EQ(rel->set_options({{"pressio:rel", rel_bound}}), 0) << rel->error_msg();

  pressio_data input = make_field<float>();
  pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
  pressio_data decompressed = pressio_data::owning(input.dtype(), input.dimensions());
  ASSERT_EQ(rel->compress(&input, &compressed), 0) << rel->error_msg();
  ASSERT_EQ(rel->decompress(&compressed, &decompressed), 0) << rel->error_msg();

  float const* values = static_cast<float const*>(input.data());
  auto range = std::minmax_element(values, values + input.num_elements());
  double const value_range = static_cast<double>(*range.second) - static_cast<double>(*range.first);
  EXPECT_LE(max_abs_error<float>(input, decompressed), rel_bound * value_range * 1.01);
}

TEST_F(SZGPUPlugin, ConcurrentCallsOnOneInstance) {
  pressio_data input = make_field<float>();
  std::vector<double> errors(4, 0.0);
  std::vector<std::thread> threads;
  for (size_t t = 0; t < errors.size(); ++t) {
    threads.emplace_back([&, t] {
      for (int iteration = 0; iteration < 8; ++iteration) {
        pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
        pressio_data decompressed = pressio_data::owning(input.dtype(), input.dimensions());
        if (compressor->compress(&input, &compressed) || compressor->decompress(&compressed, &decompressed)) {
          errors[t] = std::numeric_limits<double>::infinity();
          return;
        }
        errors[t] = std::max(errors[t], max_abs_error<float>(input, decompressed));
      }
    });
  }
  for (auto& thread : threads) thread.join();
  for (double error : errors) EXPECT_LE(error, abs_bound * 1.01);
}

TEST_F(SZGPUPlugin, ReportsErrors) {
  pressio_data integers = pressio_data::owning(pressio_int32_dtype, dims);
  pressio_data compressed = pressio_data::empty(pressio_byte_dtype, {});
  EXPECT_NE(compressor->compress(&integers, &compressed), 0);

  pressio_data input = make_field<float>();
  ASSERT_EQ(compressor->compress(&input, &compressed), 0) << compressor->error_msg();
  pressio_data too_large = pressio_data::owning(pressio_float_dtype, {100 * input.num_elements()});
  EXPECT_NE(compressor->decompress(&compressed, &too_large), 0);

  ASSERT_EQ(compressor->set_options({{"pressio:abs", 0.0}}), 0) << compressor->error_msg();
  EXPECT_NE(compressor->compress(&input, &compressed), 0);
}
