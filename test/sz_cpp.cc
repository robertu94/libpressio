#include <iostream>
#include <libpressio_ext/cpp/libpressio.h>
#include <sz.h>

#include "make_input_data.h"

int main()
{
  pressio library;
  auto compressor = library.get_compressor("sz");
  const char* metrics_ids[] = {"time"};
  auto metrics = pressio_metrics(library.get_metrics(std::begin(metrics_ids), std::end(metrics_ids)));

  compressor->set_metrics(metrics);
  pressio_options options = compressor->get_options();
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
  std::vector<size_t> dims{300,300,300};

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
  for (auto const& metric : metrics_results) {
    std::cout << metric.second << std::endl;
  }

  return 0;
}
