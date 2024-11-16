#include <iostream>
#include <ios>
#include <cmath>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include <functional>
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "std_compat/memory.h"

namespace libpressio { namespace out_of_bounds_metrics_ns {

class out_of_bounds_plugin : public libpressio_metrics_plugin {
  public:
    int end_compress_impl(struct pressio_data const* real_input, pressio_data const*, int) override {
      in =  pressio_data::clone(domain_manager().make_readable(domain_plugins().build("malloc"), *real_input));
      return 0;
    }

    int end_decompress_impl(struct pressio_data const*, pressio_data const* real_output, int) override {
      std::function<std::pair<bool, double>(double,double)> compare;
      switch(mode) {
          case bound_type::abs:
              compare = [this](double x, double y){ return std::make_pair(std::fabs(x-y)>bound,std::fabs(x-y)); };
              break;
          case bound_type::rel:
              {
                  double range = pressio_data_for_each<double>((pressio_data const&)in, [](auto* i, auto* j) -> double { auto mm = std::minmax_element(i, j); return *mm.second - *mm.first; });
                  compare = [this,range](double x, double y){ return std::make_pair(std::fabs(x-y)/range>bound,std::fabs(x-y)/range); };
              }
              break;
          case bound_type::pw_rel:
              compare = [this](double x, double y){ return std::make_pair((std::fabs(x-y)/x)>bound,std::fabs(x-y)/x); };
              break;
      }


      auto out = domain_manager().make_readable(domain_plugins().build("malloc"), *real_output);
      std::tie(oob, index) = pressio_data_for_each<std::pair<pressio_data,pressio_data>>((pressio_data const&)in, (pressio_data const&)out, [&compare, this](auto in_begin, auto in_end, auto out_begin, auto out_end){
              //determine input size
              const size_t in_N = std::distance(in_begin, in_end);
              const size_t out_N = std::distance(out_begin, out_end);
              const size_t N = std::min(in_N, out_N);
              size_t num_prints = print_first_k;
              bool should_print_exceeds = print_first_k != 0;

              //find errors
              std::vector<size_t> indexes;
              std::vector<std::tuple<double, double, double>> out_of_bounds;
              for(size_t i= 0; i < N; ++i) {
                const auto [out_of_range, error] = compare((double)in_begin[i], (double)out_begin[i]);
                if(out_of_range) {
                    out_of_bounds.emplace_back(in_begin[i], out_begin[i], error);
                    indexes.emplace_back(i);
                    if(num_prints > 0) {
                        num_prints--;
                        std::cerr << "u[i=" << i << "]" << "=" << std::scientific << (double)in_begin[i] << " d[i]=" << (double) out_begin[i] << " e[i]=" << error << "\n";
                    }
                }
              }

              if(should_print_exceeds && num_prints == 0) {
                    std::cerr << "exceeded num_prints" << std::endl;
              }
              
              //copy to pressio_data
              auto oob = pressio_data::owning(pressio_double_dtype, {3, out_of_bounds.size()});
              double* oob_ptr = static_cast<double*>(oob.data());
              auto index = pressio_data::owning(pressio_uint64_dtype, {out_of_bounds.size()});
              uint64_t* index_ptr = static_cast<uint64_t*>(index.data());
              for(size_t i= 0; i < out_of_bounds.size(); ++i) {
                  std::tie(oob_ptr[i*3], oob_ptr[i*3+1], oob_ptr[i*3+2]) = out_of_bounds[i];
                  index_ptr[i] = indexes[i];
              }

              //return final result
              return std::make_pair(std::move(oob), std::move(index));
              
      });
      return 0;
    }

  pressio_options get_options() const override {
      pressio_options opts;
      set(opts, "out_of_bounds:print_first_k", print_first_k);
      set(opts, "pressio:abs", (mode == bound_type::abs) ? compat::optional<double>{}: compat::optional<double>{bound});
      set(opts, "pressio:rel", (mode == bound_type::rel) ? compat::optional<double>{}: compat::optional<double>{bound});
      set(opts, "pressio:pw_rel", (mode == bound_type::rel) ? compat::optional<double>{}: compat::optional<double>{bound});
      return opts;
  }

  int set_options(pressio_options const& opts) override {
      get(opts, "out_of_bounds:print_first_k", &print_first_k);
      double tmp_bound;
      if(get(opts, "pressio:abs", &tmp_bound)== pressio_options_key_set) {
        bound = tmp_bound;
        mode = bound_type::abs;
      }
      if(get(opts, "pressio:rel", &tmp_bound) == pressio_options_key_set) {
        bound = tmp_bound;
        mode = bound_type::rel;
      }
      if(get(opts, "pressio:pw_rel", &tmp_bound) == pressio_options_key_set) {
        bound = tmp_bound;
        mode = bound_type::pw_rel;
      }
      return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "experimental");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", R"(produces a report of the values that are out of bounds)");
    set(opt, "out_of_bounds:oob", "a 3xN array where [i*3]= input, [i*3+1]=output, [i*3+2]=error");
    set(opt, "out_of_bounds:index", "a length N array containing the indexes of the out of range values");
    set(opt, "out_of_bounds:print_first_k",
        "print the first k entries that are out of range in a human-focused unstable format for "
        "debugging");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt, "out_of_bounds:oob", oob);
    set(opt, "out_of_bounds:index", index);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<out_of_bounds_plugin>(*this);
  }
  const char* prefix() const override {
    return "out_of_bounds";
  }

  private:
  pressio_data in, oob, index;
  uint64_t print_first_k = 0;
  double bound;
  enum class bound_type {
      abs,
      pw_rel,
      rel
  } mode;

};

static pressio_register metrics_out_of_bounds_plugin(metrics_plugins(), "out_of_bounds", [](){ return compat::make_unique<out_of_bounds_plugin>(); });
}}

