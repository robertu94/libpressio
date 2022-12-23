#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"
#include "std_compat/functional.h"
#include <numeric>

#include "basic_indexer.h"

namespace libpressio { namespace data_gap_metrics_ns {
//assume dims are fastest to slowest

    using namespace utilities;

class data_gap_plugin : public libpressio_metrics_plugin {
  public:
    struct gap_stats {
      double min = std::numeric_limits<double>::max();
      double max = std::numeric_limits<double>::lowest();
      double mean = 0.0;
    };
    struct data_gap{
      template <class T> gap_stats operator()(T const* begin, T const*) {
        switch(dims.size()) {
          case 1:
            return gap1d(begin);
          case 2:
            return gap2d(begin);
          case 3:
            return gap3d(begin);
          default:
            throw std::domain_error("data gap is only supported on 1-3d");
        }
      }
      template <class T> gap_stats gap1d(T const* begin) {
        gap_stats stats;
        T next = begin[1];
        for (size_t i = 0; i < dims[0] - 1; ++i) {
          auto gap = std::abs(static_cast<double>(begin[i] - next));
          stats.max = std::max<double>(stats.max, gap);
          stats.min = std::min<double>(stats.min, gap);
          stats.mean += gap;
          next = begin[i+1];
        }
        stats.mean /= static_cast<double>(dims[0] - 1);
        return stats;
      }
      template <class T> gap_stats gap2d(T const* begin) {
        gap_stats stats;
        indexer<2> idx{dims[0], dims[1]};
        T right = begin[idx(1,0)];
        T down = begin[idx(0,1)];
        for (size_t j = 0; j < dims[1] - 1; ++j) {
        for (size_t i = 0; i < dims[0] - 1; ++i) {
          auto current = begin[idx(i,j)];
          {
            auto gap = std::abs(static_cast<double>(current - right));
            stats.max = std::max<double>(stats.max, gap);
            stats.min = std::min<double>(stats.min, gap);
            stats.mean += gap;
          }
          {
            auto gap = std::abs(static_cast<double>(current - down));
            stats.max = std::max<double>(stats.max, gap);
            stats.min = std::min<double>(stats.min, gap);
            stats.mean += gap;
          }
          right = begin[idx(i+1,j)];
          down = begin[idx(i,j+1)];
        } }
        stats.mean /= static_cast<double>(2* (dims[0] - 1) * (dims[1] - 1));
        return stats;
      }
      template <class T> gap_stats gap3d(T const* begin) {
        gap_stats stats;
        indexer<3> idx{dims[0], dims[1], dims[2]};
        T right = begin[idx(1,0,0)];
        T down = begin[idx(0,1,0)];
        T out = begin[idx(0,0,1)];
        for (size_t k = 0; k < dims[2] - 1; ++k) {
        for (size_t j = 0; j < dims[1] - 1; ++j) {
        for (size_t i = 0; i < dims[0] - 1; ++i) {
          auto current = begin[idx(i,j,k)];
          {
            auto gap = std::abs(static_cast<double>(current - right));
            stats.max = std::max<double>(stats.max, gap);
            stats.min = std::min<double>(stats.min, gap);
            stats.mean += gap;
          }
          {
            auto gap = std::abs(static_cast<double>(current - down));
            stats.max = std::max<double>(stats.max, gap);
            stats.min = std::min<double>(stats.min, gap);
            stats.mean += gap;
          }
          {
            auto gap = std::abs(static_cast<double>(current - out));
            stats.max = std::max<double>(stats.max, gap);
            stats.min = std::min<double>(stats.min, gap);
            stats.mean += gap;
          }
          right = begin[idx(i+1,j,k)];
          down = begin[idx(i,j+1,k)];
          out = begin[idx(i,j,k+1)];
        } } }
        stats.mean /= static_cast<double>(3* (dims[0] - 1) * (dims[1] - 1) * (dims[2] -1));
        return stats;
      }
      std::vector<size_t> const& dims;
    };

    int end_compress_impl(struct pressio_data const* input, pressio_data const*, int) override {
      try {
        gap = pressio_data_for_each<gap_stats>(*input, data_gap{input->dimensions()});
        return 0;
      } catch( std::exception const& ex ) {
        return set_error(1, ex.what());
      }
    }


  
  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "computes statistics about the gaps between adjacent values");
      set(opt, "data_gap:max_gap", "the maximum gap");
      set(opt, "data_gap:min_gap", "the minimum gap");
      set(opt, "data_gap:mean_gap", "the mean gap");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    if(gap) {
      set(opt, "data_gap:max_gap", gap->max);
      set(opt, "data_gap:min_gap", gap->min);
      set(opt, "data_gap:mean_gap", gap->mean);
    } else {
      set_type(opt, "data_gap:max_gap", pressio_option_double_type);
      set_type(opt, "data_gap:min_gap", pressio_option_double_type);
      set_type(opt, "data_gap:mean_gap", pressio_option_double_type);
    }
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<data_gap_plugin>(*this);
  }
  const char* prefix() const override {
    return "data_gap";
  }

  private:
  compat::optional<gap_stats> gap;

};

static pressio_register metrics_data_gap_plugin(metrics_plugins(), "data_gap", [](){ return compat::make_unique<data_gap_plugin>(); });
}}
