#include <cmath>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "std_compat/memory.h"

namespace libpressio {
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
    double min_pw_rel_error;
    double max_pw_rel_error;
    double average_difference;
    double average_error;
    double difference_range;
    double error_range;
    double value_min;
    double value_max;
    double value_std;
    double value_mean;
    uint64_t num_elements;
  };

  struct compute_metrics{
    template <class ForwardIt1, class ForwardIt2>
    metrics operator()(ForwardIt1 input_begin, ForwardIt1 input_end, ForwardIt2 input2_begin, ForwardIt2 input2_end)
    {
      metrics m{};
      double sum_of_squared_error = 0;
      double sum_of_difference = 0;
      double sum_of_error = 0;
      double sum_of_values_squared =0;
      double sum = 0;
      double min_pw_rel_error = std::numeric_limits<double>::max();
      double max_pw_rel_error = std::numeric_limits<double>::lowest();
      if(input_begin != nullptr && input2_begin != nullptr) {
        double value_min = *input_begin; 
        double value_max = *input_begin;
        double diff_min = *input_begin - *input2_begin;
        double diff_max = diff_min;
        double error_min = std::abs(double(*input_begin) - double(*input2_begin));
        double error_max = std::abs(double(diff_min));
        while(input_begin != input_end && input2_begin != input2_end) {
          double diff = *input_begin - *input2_begin;
          double error = std::abs(double(diff));
          double squared_error = error*error;

          sum += *input_begin;
          sum_of_values_squared += (*input_begin * *input_begin);
          sum_of_difference += diff;
          sum_of_error += error;
          sum_of_squared_error += squared_error;
          value_min = std::min(value_min, static_cast<double>(*input_begin));
          value_max = std::max(value_max, static_cast<double>(*input_begin));
          diff_min = std::min(diff, diff_min);
          diff_max = std::max(diff, diff_max);
          diff_max = std::max(diff, diff_max);
          error_max = std::max(error, error_max);
          error_min = std::min(error, error_min);
          if (*input_begin != 0) {
            double pw_rel_error = std::abs(double(diff)/(*input_begin));
            min_pw_rel_error = std::min(min_pw_rel_error, pw_rel_error);
            max_pw_rel_error = std::max(max_pw_rel_error, pw_rel_error);
          }

          ++m.num_elements;

          ++input_begin;
          ++input2_begin;
        }
        m.mse = sum_of_squared_error/static_cast<double>(m.num_elements);
        m.rmse = sqrt(m.mse);
        m.average_difference = sum_of_difference/static_cast<double>(m.num_elements);
        m.average_error = sum_of_error/static_cast<double>(m.num_elements);

        m.value_min = value_min;
        m.value_max = value_max;
        m.value_mean = sum/static_cast<double>(m.num_elements);
        m.value_std = std::sqrt((sum_of_values_squared - ((sum*sum)/static_cast<double>(m.num_elements))) / static_cast<double>(m.num_elements));
        m.value_range = value_max-value_min;

        m.difference_range = diff_max - diff_min;
        m.error_range = error_max - error_min;

        m.min_error = error_min;
        m.max_error = error_max;
        m.min_rel_error = error_min/m.value_range;
        m.max_rel_error = error_max/m.value_range;
        m.min_pw_rel_error = min_pw_rel_error;
        m.max_pw_rel_error = max_pw_rel_error;

        m.psnr = -20.0*log10(sqrt(m.mse)/m.value_range);
      }

      return m;
    }
  };

class error_stat_plugin : public libpressio_metrics_plugin {

  public:
    int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * ) override {
      input_data = pressio_data::clone(*input);
      return 0;
    }
    int end_decompress_impl(struct pressio_data const*, struct pressio_data const* output, int ) override {
      err_metrics = pressio_data_for_each<error_stat::metrics>(input_data, *output, error_stat::compute_metrics{});
      return 0;
    }

    struct pressio_options get_configuration_impl() const override {
      pressio_options opts;
      set(opts, "pressio:stability", "stable");
      set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(opts, "predictors:error_agnostic", std::vector<std::string>{
              "error_stat:value_mean",
              "error_stat:value_std",
              "error_stat:value_min",
              "error_stat:value_max",
              "error_stat:value_range",
              "error_stat:n",
              });
      std::vector<std::string> reqs_dec {
          "error_stat:psnr",
              "error_stat:mse",
              "error_stat:rmse",
              "error_stat:min_error",
              "error_stat:max_error",
              "error_stat:min_rel_error",
              "error_stat:max_rel_error",
              "error_stat:min_pw_rel_error",
              "error_stat:max_pw_rel_error",
              "error_stat:average_difference",
              "error_stat:average_error",
              "error_stat:difference_range",
              "error_stat:error_range",
      };
      set(opts, "predictors:requires_decompress", reqs_dec);
      set(opts, "predictors:error_dependent", reqs_dec);
      return opts;
    }


    struct pressio_options get_documentation_impl() const override {
      pressio_options opt;
      set(opt, "pressio:description", "Basic error statistics that can be computed in in one pass");
      set(opt, "error_stat:psnr", "peak signal to noise ratio");
      set(opt, "error_stat:mse", "mean squared error");
      set(opt, "error_stat:rmse", "root mean squared error");
      set(opt, "error_stat:value_mean", "the mean of the input values");
      set(opt, "error_stat:value_std", "standard deviation of the input values");
      set(opt, "error_stat:value_min", "minimum of the input values");
      set(opt, "error_stat:value_max", "maximum of the input values");
      set(opt, "error_stat:value_range", "the range of the input values");
      set(opt, "error_stat:min_error", "the minimum absolute difference");
      set(opt, "error_stat:max_error", "the maximum absolute difference");
      set(opt, "error_stat:min_rel_error", "the minimum absolute difference relative to the input value range");
      set(opt, "error_stat:max_rel_error", "the maximum absolute difference relative to the input value range");
      set(opt, "error_stat:min_pw_rel_error", "the minimum absolute difference relative to each data point");
      set(opt, "error_stat:max_pw_rel_error", "the maximum absolute difference relative to each data point");
      set(opt, "error_stat:average_difference", "the average difference");
      set(opt, "error_stat:average_error", "the average absolute difference");
      set(opt, "error_stat:difference_range", "the range of the differences");
      set(opt, "error_stat:error_range", "the range of the absolute differences");
      set(opt, "error_stat:n", "the number of input values");
      return opt;
    }
    pressio_options get_metrics_results(pressio_options const &)  override {
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
        set(opt, "error_stat:min_pw_rel_error", (*err_metrics).min_pw_rel_error);
        set(opt, "error_stat:max_pw_rel_error", (*err_metrics).max_pw_rel_error);
        set(opt, "error_stat:average_difference", (*err_metrics).average_difference);
        set(opt, "error_stat:average_error", (*err_metrics).average_error);
        set(opt, "error_stat:difference_range", (*err_metrics).difference_range);
        set(opt, "error_stat:error_range", (*err_metrics).error_range);
        set(opt, "error_stat:n", (*err_metrics).num_elements);
      } else {
        set_type(opt, "error_stat:n", pressio_option_uint64_type);
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
        set_type(opt, "error_stat:min_pw_rel_error", pressio_option_double_type);
        set_type(opt, "error_stat:max_pw_rel_error", pressio_option_double_type);
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
}}
