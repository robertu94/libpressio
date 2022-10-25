#include <cstdint>
#include <vector>
#include <pressio_dtype.h>
#include <libpressio_ext/cpp/options.h>

struct compression_options {
  pressio_dtype dtype;
  std::vector<size_t> dims;
  std::string compressor_id;
  pressio_options options;
};

std::vector<unsigned int> get_cd_values_from_options(compression_options const& options);
