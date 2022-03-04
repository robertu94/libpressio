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
  std::vector<std::size_t> dims{500,500,100};
  std::vector<float> v(product(dims));
  std::iota(v.begin(), v.end(), 0);
  auto input = pressio_data::nonowning(pressio_float_dtype, v.data(), dims);
  auto compressed = pressio_data::empty(pressio_byte_dtype, {});
  auto decompressed = pressio_data::owning(pressio_float_dtype, dims);

  std::seed_seq seed{0};
  std::mt19937 gen{seed};
  std::uniform_int_distribution<size_t> dist_row(0,dims[0]-1);
  std::uniform_int_distribution<size_t> dist_col(0,dims[1]-1);
  std::uniform_int_distribution<size_t> dist_event(0,dims[2]-1);
  auto rnd_row = [&]{ return dist_row(gen); };
  auto rnd_col = [&]{ return dist_col(gen); };
  auto rnd_event = [&]{ return dist_event(gen); };

  constexpr std::size_t N_roi =  500;
  pressio_data centers = pressio_data::owning(pressio_uint64_dtype, {3, 500});
  uint64_t* centers_ptr = static_cast<uint64_t*>(centers.data());
  for (size_t i = 0; i < N_roi; ++i) {
    centers_ptr[3*i] = rnd_row();
    centers_ptr[3*i+1] = rnd_col();
    centers_ptr[3*i+2] = rnd_event();
  }

  pressio library;
  pressio_compressor roibin = library.get_compressor("roibin");
  pressio_options options {
    {"roibin:background", "binning"},
    {"roibin:roi", "fpzip"},
    {"roibin:nthreads", 6u},
    {"blosc:compressor", "zstd"},
    {"blosc:clevel", 6},
    {"fpzip:prec", 0},
    {"pressio:metric", "composite"},
    {"composite:plugins", std::vector<std::string>{"size", "time"}},
    {"binning:compressor", "sz"},
    {"binning:shape", pressio_data{2,2,1}},
    {"binning:nthreads", 6u},
    {"pressio:abs", 1e-4},
    {"roibin:centers", centers},
    {"roibin:roi_size", pressio_data{8,8,0}}
  };

  roibin->set_options(options);
  roibin->set_name("pressio");
  std::cout << "settings" << std::endl;
  std::cout << roibin->get_options() << std::endl;

  if(roibin->compress(&input, &compressed)) {
    std::cerr << roibin->error_msg() << std::endl;
    exit(roibin->error_code());
  }

  if(roibin->decompress(&compressed, &decompressed)) {
    std::cerr << roibin->error_msg() << std::endl;
    exit(roibin->error_code());
  }


  std::cout << "metrics" << std::endl;
  std::cout << roibin->get_metrics_results() << std::endl;
  
}
