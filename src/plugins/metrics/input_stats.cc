#include <algorithm>
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "std_compat/memory.h"

namespace libpressio { namespace metrics { namespace input_stats_metrics_ns {

  struct input_stat{
    input_stat(compat::span<const pressio_data* const> const& inputs,
               compat::span<const pressio_data* const> const& outputs):
      input_dims(dims_size(inputs)),
      output_dims(dims_size(outputs)),
      input_types(type(inputs)),
      output_types(type(outputs)) {
    }
    input_stat()=default;
    input_stat(input_stat const&)=default;
    input_stat(input_stat &&)=default;
    input_stat& operator=(input_stat const&)=default;
    input_stat& operator=(input_stat &&)=default;

    pressio_data input_dims;
    pressio_data output_dims;
    pressio_data input_types;
    pressio_data output_types;

    static pressio_data dims_size(compat::span<const pressio_data* const> const& inputs) {
      std::vector<size_t> lengths(inputs.size());
      std::transform(inputs.begin(), inputs.end(), lengths.begin(), [](const pressio_data* const data){
            return data->num_dimensions();
          });
      size_t max_length = *std::max_element(lengths.begin(), lengths.end());
      auto data = pressio_data::owning(pressio_uint64_dtype, {max_length,inputs.size()});
      uint64_t* dims_info = static_cast<uint64_t*>(data.data());
      for (size_t j = 0; j < max_length; ++j) {
        for (size_t i = 0; i < inputs.size(); ++i) {
          dims_info[i*max_length+j] = inputs[i]->get_dimension(j);
        }
      }
      return data;
    }
    static pressio_data type(compat::span<const pressio_data* const> const& inputs) {
      using underlying_type = typename std::underlying_type<pressio_dtype>::type;
      auto data = pressio_data::owning(
          pressio_dtype_from_type<underlying_type>(),
          {inputs.size()}
          );
      underlying_type* ptr = static_cast<underlying_type*>(data.data());
      for (size_t i = 0; i < inputs.size(); ++i) {
        ptr[i] = static_cast<underlying_type>(inputs[i]->dtype());  
      }


      return data;
    }
  };

class input_stats_plugin : public libpressio_metrics_plugin {
  public:
    int end_compress_impl(struct pressio_data const* input, pressio_data const* output, int rc) override {
      if(!input || !input->has_data()) return 0;
      if(!output || !output->has_data()) return 0;
      compat::span<const pressio_data*> inputs(&input, 1);
      compat::span<const pressio_data*> outputs(&output, 1);
      return end_compress_many_impl(inputs, outputs, rc);
    }

    int end_decompress_impl(struct pressio_data const* input, pressio_data const* output, int rc) override {
      if(!input || !input->has_data()) return 0;
      if(!output || !output->has_data()) return 0;
      compat::span<const pressio_data*> inputs(&input, 1);
      compat::span<const pressio_data*> outputs(&output, 1);
      return end_decompress_many_impl(inputs, outputs, rc);
    }

    int end_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int ) override {
      compression = input_stat(inputs, outputs);
      return 0;
  }

  int end_decompress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const& outputs, int ) override {
    decompression = input_stat(inputs, outputs);
    return 0;
  }

  
  struct pressio_options get_configuration_impl() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(opts, "predictors:requires_decompress", true);
    set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:data"});
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set(opt, "pressio:description", "records the sizes and types of inputs");
    set(opt,"input_stat:compress_input_dims", "sizes of compress inputs");
    set(opt,"input_stat:compress_output_dims", "sizes of compress outputs");
    set(opt,"input_stat:decompress_input_dims", "sizes of decompress inputs");
    set(opt,"input_stat:decompress_output_dims", "sizes of decompress outputs");
    set(opt,"input_stat:compress_input_types", "types of compression inputs");
    set(opt,"input_stat:compress_output_types", "types of compression outputs");
    set(opt,"input_stat:decompress_input_types", "types of decompression inputs");
    set(opt,"input_stat:decompress_output_types", "types of decompression outputs");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    set(opt,"input_stat:compress_input_dims", compression.input_dims);
    set(opt,"input_stat:compress_output_dims", compression.output_dims);
    set(opt,"input_stat:decompress_input_dims", decompression.input_dims);
    set(opt,"input_stat:decompress_output_dims", decompression.output_dims);
    set(opt,"input_stat:compress_input_types", compression.input_types);
    set(opt,"input_stat:compress_output_types", compression.output_types);
    set(opt,"input_stat:decompress_input_types", decompression.input_types);
    set(opt,"input_stat:decompress_output_types", decompression.output_types);
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<input_stats_plugin>(*this);
  }
  const char* prefix() const override {
    return "input_stats";
  }

  private:
  input_stat compression;
  input_stat decompression;
};

pressio_register registration(metrics_plugins(), "input_stats", [](){ return compat::make_unique<input_stats_plugin>(); });
}}
}
