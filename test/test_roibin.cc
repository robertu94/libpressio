#include <libpressio_ext/cpp/pressio.h>
#include <libpressio_ext/cpp/printers.h>
#include <std_compat/functional.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <random>

template <class Rng>
auto product(Rng const& rng) -> typename std::iterator_traits<decltype(rng.begin())>::value_type {
  return std::accumulate(rng.begin(), rng.end(), 1, compat::multiplies<>{});
}

int main()
{
  std::vector<std::size_t> dims{500,500,100,4};
  std::vector<float> v(product(dims));
  std::iota(v.begin(), v.end(), 0);
  auto input = pressio_data::nonowning(pressio_float_dtype, v.data(), dims);
  auto compressed = pressio_data::empty(pressio_byte_dtype, {});
  auto decompressed = pressio_data::owning(pressio_float_dtype, dims);

  std::seed_seq seed{0};
  std::mt19937 gen{seed};
  std::uniform_int_distribution<size_t> dist_row(0,dims[0]-1);
  std::uniform_int_distribution<size_t> dist_col(0,dims[1]-1);
  std::uniform_int_distribution<size_t> dist_seg(0,dims[2]-1);
  auto rnd_row = [&]{ return dist_row(gen); };
  auto rnd_col = [&]{ return dist_col(gen); };
  auto rnd_seg = [&]{ return dist_seg(gen); };

  std::vector<size_t> rows, cols, segs;
  constexpr std::size_t N_roi =  500;
  std::generate_n(std::back_inserter(rows), N_roi, rnd_row);
  std::generate_n(std::back_inserter(cols), N_roi, rnd_col);
  std::generate_n(std::back_inserter(segs), N_roi, rnd_seg);

  pressio library;
  pressio_compressor roibin = library.get_compressor("roibin");
  pressio_options options {
    {"roibin:background", "binning"},
    {"roibin:roi", "fpzip"},
    {"pressio:metric", "composite"},
    {"composite:plugins", std::vector<std::string>{"size", "time"}},
    {"binning:compressor", "sz"},
    {"pressio:abs", 1e-4},
    {"roibin:rows", pressio_data(rows.begin(), rows.end())},
    {"roibin:cols", pressio_data(cols.begin(), cols.end())},
    {"roibin:segs", pressio_data(segs.begin(), segs.end())}
  };

  roibin->set_options(options);
  roibin->set_name("pressio");
  std::cout << "settings" << std::endl;
  std::cout << roibin->get_options() << std::endl;

  roibin->compress(&input, &compressed);
  roibin->decompress(&compressed, &decompressed);


  std::cout << "metrics" << std::endl;
  std::cout << roibin->get_metrics_results() << std::endl;
  
}
