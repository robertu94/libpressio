#include <iostream>
#include <iomanip>
#include <sstream>
#include <cstdint>
#include <set>
#include <cmath>
#include <libpressio_ext/cpp/libpressio.h>
#include <pressio_version.h>

void lp_assert(bool is_correct) {
  if(!is_correct) {
    std::cout << "failed" << std::endl;
    exit(1);
  }
}

int main(int argc, char * argv[])
{
  std::set<std::string> skip;
  int run_stability = 0;
  if (argc >= 2) {
    run_stability = atoi(argv[1]);
  }
  for (int i = 2; i < argc; ++i) {
   skip.emplace(argv[i]);
  }


  pressio library;
  lp_assert(LIBPRESSIO_MAJOR_VERSION == library.major_version());
  lp_assert(LIBPRESSIO_MINOR_VERSION == library.minor_version());
  lp_assert(LIBPRESSIO_PATCH_VERSION == library.patch_version());


  struct {
    int a;
    int b;
  } s;
  void* struct_ptr = &s;

  pressio_options opts;
  opts.set("i8", int8_t{1});
  opts.set("i16", int16_t{2});
  opts.set("i32", int32_t{3});
  opts.set("i64", int64_t{4});
  opts.set("u8", uint8_t{5});
  opts.set("u16", uint16_t{6});
  opts.set("u32", uint32_t{7});
  opts.set("u64", uint64_t{8});
  opts.set("f32", float{9.0f});
  opts.set("f64", float{10.0});
  opts.set("bool", true);
  opts.set("void", struct_ptr);
  opts.set("data", pressio_data::owning(pressio_float_dtype, {1}));

  int8_t i8;
  lp_assert(opts.get("i8", &i8) == pressio_options_key_set);
  lp_assert(i8 == 1);
  int16_t i16;
  lp_assert(opts.get("i16", &i16) == pressio_options_key_set);
  lp_assert(i16 == 2);
  int32_t i32;
  lp_assert(opts.get("i32", &i32) == pressio_options_key_set);
  lp_assert(i32 == 3);
  int64_t i64;
  lp_assert(opts.get("i64", &i64) == pressio_options_key_set);
  lp_assert(i64 == 4);
  uint8_t u8;
  lp_assert(opts.get("u8", &u8) == pressio_options_key_set);
  lp_assert(u8 == 5);
  uint16_t u16;
  lp_assert(opts.get("u16", &u16) == pressio_options_key_set);
  lp_assert(u16 == 6);
  uint32_t u32;
  lp_assert(opts.get("u32", &u32) == pressio_options_key_set);
  lp_assert(u32 == 7);
  uint64_t u64;
  lp_assert(opts.get("u64", &u64) == pressio_options_key_set);
  lp_assert(u64 == 8);
  float f32;
  lp_assert(opts.get("f32", &f32) == pressio_options_key_set);
  lp_assert(f32 == 9.0f);
  float f64;
  lp_assert(opts.get("f64", &f64) == pressio_options_key_set);
  lp_assert(f64 == 10.0);
  bool b;
  lp_assert(opts.get("bool", &b) == pressio_options_key_set);
  lp_assert(b == true);
  void* v;
  lp_assert(opts.get("void", &v) == pressio_options_key_set);
  lp_assert(v == struct_ptr);
  pressio_data d;
  lp_assert(opts.get("data", &d) == pressio_options_key_set);
  lp_assert(d.dtype() == pressio_float_dtype);
  lp_assert(d.dimensions() == std::vector<size_t>{1});


  std::vector<size_t> dims{size_t{200}, size_t{200}};
  pressio_data input(pressio_data::owning(pressio_float_dtype, dims));
  for (size_t i = 0; i < dims[0]; ++i) {
    for (size_t j = 0; j < dims[1]; ++j) {
      double x = static_cast<double>(i) - static_cast<double>(dims[0])/2.0;
      double y = static_cast<double>(j)-  static_cast<double>(dims[1])/2.0;
      *static_cast<float*>(input.data()) = static_cast<float>(.0001* y * sin(y) + .0005 * cos(pow(x,2) + x));
    }
  }
  pressio_data compressed(pressio_data::empty(pressio_byte_dtype, {}));
  pressio_data output(pressio_data::owning(input.dtype(), input.dimensions()));

  std::istringstream supported_compressors(library.supported_compressors());
  for (std::string compressor_id; std::getline(supported_compressors, compressor_id, ' '); ) {
    pressio_compressor comp = library.get_compressor(compressor_id);
    lp_assert(comp);

    pressio_options options = comp->get_options();
    pressio_options configuration = comp->get_configuration();
    std::string stability;
    lp_assert(configuration.get("pressio:stability", &stability) == pressio_options_key_set);
    lp_assert(comp->set_options({{"pressio:metric", "error_stat"}}) == 0);
    std::cout << compressor_id << " passed: init" << std::endl;
    if(skip.find(compressor_id) != skip.end()) {
      std::cout << "skipping " << compressor_id << std::endl;
      continue;
    }

    if((stability == "stable") || (run_stability>=1 && stability=="unstable") || (run_stability >=2 && stability=="experimental") ) {

      if(options.key_status("pressio:abs") != pressio_options_key_does_not_exist) {
        for (double bound : {1e-5, 1e-4, 1e-3}) {
          comp->set_options(pressio_options{{"pressio:abs", bound}});
          lp_assert(comp->compress(&input, &compressed) == 0);
          lp_assert(comp->decompress(&compressed, &output) == 0);
          auto metrics = comp->get_metrics_results();
          double max_error;
          lp_assert(metrics.get("error_stat:max_error", &max_error) == pressio_options_key_set);
          lp_assert(max_error <= bound);
        }
        std::cout << compressor_id << " passed: abs" << std::endl;
      }

      if(options.key_status("pressio:lossless") != pressio_options_key_does_not_exist) {
        comp->set_options(pressio_options{{"pressio:lossless", int32_t{1}}});
        lp_assert(comp->compress(&input, &compressed) == 0);
        lp_assert(comp->decompress(&compressed, &output) == 0);
        auto metrics = comp->get_metrics_results();
        double max_error;
        lp_assert(metrics.get("error_stat:max_error", &max_error) == pressio_options_key_set);
        lp_assert(max_error <= 0);
        std::cout << compressor_id << " passed: lossless" << std::endl;
      }
    }
  }

  
  std::cout << "all passed" << std::endl;
  return 0;
}
