#include <cmath>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/memory.h"

namespace error_stat {
  struct metrics {
    double psnr;
    double mse;
    double rmse;
    double value_range;
    double min_error;
    double max_error;
    double min_rel_error;
    double max_rel_error;
    double average_difference;
    double average_error;
    double difference_range;
    double error_range;
    double value_min;
    double value_max;
    double value_std;
    double value_mean;
  };

  struct compute_metrics{
    template <class ForwardIt1, class ForwardIt2>
    metrics operator()(ForwardIt1 input_begin, ForwardIt1 input_end, ForwardIt2 input2_begin, ForwardIt2 input2_end)
    {
      using value_type = typename std::iterator_traits<ForwardIt1>::value_type;
      static_assert(std::is_same<typename std::iterator_traits<ForwardIt1>::value_type, value_type>::value, "the iterators must have the same type");
      double sum_of_squared_error = 0;
      double sum_of_difference = 0;
      double sum_of_error = 0;
      double sum_of_values_squared =0;
      double sum = 0;
      size_t num_elements = 0;
      auto value_min = *input_begin; 
      auto value_max = *input_begin;
      auto diff_min = *input_begin - *input2_begin;
      auto diff_max = diff_min;
      auto error_min = std::abs(double(*input_begin) - double(*input2_begin));
      auto error_max = std::abs(double(diff_min));
      while(input_begin != input_end && input2_begin != input2_end) {
        auto diff = *input_begin - *input2_begin;
        auto error = std::abs(double(diff));
        auto squared_error = error*error;

        sum += *input_begin;
        sum_of_values_squared += (*input_begin * *input_begin);
        sum_of_difference += diff;
        sum_of_error += error;
        sum_of_squared_error += squared_error;
        value_min = std::min(value_min, *input_begin);
        value_max = std::max(value_max, *input_begin);
        diff_min = std::min(diff, diff_min);
        diff_max = std::max(diff, diff_max);
        error_max = std::max(error, error_max);
        error_min = std::min(error, error_min);
        ++num_elements;

        ++input_begin;
        ++input2_begin;
      }
      metrics m;
      m.mse = sum_of_squared_error/num_elements;
      m.rmse = sqrt(m.mse);
      m.average_difference = sum_of_difference/num_elements;
      m.average_error = sum_of_error/num_elements;

      m.value_min = value_min;
      m.value_max = value_max;
      m.value_mean = sum/static_cast<double>(num_elements);
      m.value_std = std::sqrt((sum_of_values_squared - ((sum*sum)/static_cast<double>(num_elements))) / static_cast<double>(num_elements));
      m.value_range = value_max-value_min;

      m.difference_range = diff_max - diff_min;
      m.error_range = error_max - error_min;

      m.min_error = error_min;
      m.max_error = error_max;
      m.min_rel_error = error_min/m.value_range;
      m.max_rel_error = error_max/m.value_range;

      m.psnr = -20.0*log10(sqrt(m.mse)/m.value_range);

      return m;
    }
  };
}

class error_stat_plugin : public libpressio_metrics_plugin {

  public:
    void begin_compress(const struct pressio_data * input, struct pressio_data const * ) override {
      input_data = pressio_data::clone(*input);
    }
    void end_decompress(struct pressio_data const*, struct pressio_data const* output, int ) override {
      err_metrics = pressio_data_for_each<error_stat::metrics>(input_data, *output, error_stat::compute_metrics{});      
    }

    struct pressio_options get_metrics_results() const override {
      pressio_options opt;
      if(err_metrics) {
        set(opt, "error_stat:psnr", (*err_metrics).psnr);
        set(opt, "error_stat:mse", (*err_metrics).mse);
        set(opt, "error_stat:rmse", (*err_metrics).rmse);
        set(opt, "error_stat:value_mean", (*err_metrics).value_mean);
        set(opt, "error_stat:value_std", (*err_metrics).value_std);
        set(opt, "error_stat:value_min", (*err_metrics).value_min);
        set(opt, "error_stat:value_max", (*err_metrics).value_max);
        set(opt, "error_stat:value_range", (*err_metrics).value_range);
        set(opt, "error_stat:min_error", (*err_metrics).min_error);
        set(opt, "error_stat:max_error", (*err_metrics).max_error);
        set(opt, "error_stat:min_rel_error", (*err_metrics).min_rel_error);
        set(opt, "error_stat:max_rel_error", (*err_metrics).max_rel_error);
        set(opt, "error_stat:average_difference", (*err_metrics).average_difference);
        set(opt, "error_stat:average_error", (*err_metrics).average_error);
        set(opt, "error_stat:difference_range", (*err_metrics).difference_range);
        set(opt, "error_stat:error_range", (*err_metrics).error_range);
      } else {
        set_type(opt, "error_stat:psnr", pressio_option_double_type);
        set_type(opt, "error_stat:mse", pressio_option_double_type);
        set_type(opt, "error_stat:rmse", pressio_option_double_type);
        set_type(opt, "error_stat:value_mean", pressio_option_double_type);
        set_type(opt, "error_stat:value_std", pressio_option_double_type);
        set_type(opt, "error_stat:value_min", pressio_option_double_type);
        set_type(opt, "error_stat:value_max", pressio_option_double_type);
        set_type(opt, "error_stat:value_range", pressio_option_double_type);
        set_type(opt, "error_stat:min_error", pressio_option_double_type);
        set_type(opt, "error_stat:max_error", pressio_option_double_type);
        set_type(opt, "error_stat:min_rel_error", pressio_option_double_type);
        set_type(opt, "error_stat:max_rel_error", pressio_option_double_type);
        set_type(opt, "error_stat:average_difference", pressio_option_double_type);
        set_type(opt, "error_stat:average_error", pressio_option_double_type);
        set_type(opt, "error_stat:difference_range", pressio_option_double_type);
        set_type(opt, "error_stat:error_range", pressio_option_double_type);
      }
      return opt;
    }
    std::unique_ptr<libpressio_metrics_plugin> clone() override {
      return compat::make_unique<error_stat_plugin>(*this);
    }

  const char* prefix() const override {
    return "error_stat";
  }


  private:
  pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
  compat::optional<error_stat::metrics> err_metrics;

};

static pressio_register metrics_error_stat_plugin(metrics_plugins(), "error_stat", [](){ return compat::make_unique<error_stat_plugin>(); });
