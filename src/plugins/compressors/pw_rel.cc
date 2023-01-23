#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include <cmath>

namespace libpressio { namespace pw_rel_ns {

  struct compressor {
    template <class T>
    pressio_data operator()(T* start, T* end) {
      pressio_data signs = pressio_data::owning(pressio_bool_dtype, input.dimensions());
      pressio_data logs = pressio_data::owning(input.dtype(), input.dimensions());
      T* logs_ptr = static_cast<T*>(logs.data());
      size_t size = std::distance(start, end);

      T min, max;
      min = max = start[0];
      for (size_t i = 0; i < size; ++i) {
        min = std::min(start[i], min);
        max = std::max(start[i], max);
      }
      T max_abs_log_data = 0;
      if(min == 0) max_abs_log_data = fabs(log2(fabs(max)));
      else if(max == 0) max_abs_log_data = fabs(log2(fabs(min)));
      else max_abs_log_data = std::max(fabs(log2(fabs(min))), fabs(log2(fabs(max))));
      T min_log_data = max_abs_log_data;
      bool save_signs = false;
      bool* signs_ptr = static_cast<bool*>(signs.data());

      for (size_t i = 0; i < size; ++i) {
        if(start[i] < 0) {
          signs_ptr[i] = true;
          save_signs = true;
          logs_ptr[i] = -start[i];
        } else {
          signs_ptr[i] = false;
          logs_ptr[i] = start[i];
        }

        if(logs_ptr[i] > 0) {
          logs_ptr[i] = log2(logs_ptr[i]);
          if(logs_ptr[i] > max_abs_log_data) max_abs_log_data = logs_ptr[i];
          if(logs_ptr[i] < min_log_data) min_log_data = logs_ptr[i];
        }
      }
      if(fabs(min_log_data) > max_abs_log_data) max_abs_log_data = fabs(min_log_data);
      double abs_bound = log2(pw_rel + 1.0) - max_abs_log_data * std::numeric_limits<T>::epsilon();
      double zero_flag = min_log_data - 2.0001*abs_bound;
      double threshold = min_log_data - 1.0001*abs_bound;
      for (size_t i = 0; i < size; ++i) {
        if(start[i] == 0.0) {
          logs_ptr[i] = zero_flag;
        }
      }

      pressio_data logs_comp = pressio_data::empty(pressio_byte_dtype, {});
      pressio_data signs_comp = pressio_data::empty(pressio_byte_dtype, {});
      abs_comp->set_options({
          {"pressio:abs", abs_bound}
      });
      if(abs_comp->compress(&logs, &logs_comp) > 0) {
        throw std::runtime_error(abs_comp->error_msg());
      }
      if(save_signs) {
        if(sign_comp->compress(&signs, &signs_comp) > 0) {
          throw std::runtime_error(sign_comp->error_msg());
        }
      }

      pressio_data compressed;
      if(save_signs) {
        compressed = pressio_data::owning(pressio_byte_dtype,
            {signs_comp.size_in_bytes() + 2* sizeof(size_t) + sizeof(double) + logs_comp.size_in_bytes()});
        auto compressed_ptr_size_t = static_cast<size_t*>(compressed.data());
        auto compressed_ptr_double = reinterpret_cast<double*>(compressed_ptr_size_t + 2);
        auto compressed_ptr = reinterpret_cast<unsigned char*>(compressed_ptr_double+1);
        compressed_ptr_size_t[0] = logs_comp.size_in_bytes();
        compressed_ptr_size_t[1] = signs_comp.size_in_bytes();
        compressed_ptr_double[0] = threshold;
        memcpy(compressed_ptr, logs_comp.data(), logs_comp.size_in_bytes());
        memcpy(compressed_ptr+logs_comp.size_in_bytes(), signs_comp.data(), signs_comp.size_in_bytes());
      } else {
        compressed = pressio_data::owning(pressio_byte_dtype,
            {logs_comp.size_in_bytes() + 2*sizeof(size_t) + sizeof(double)});
        auto compressed_ptr_size_t = static_cast<size_t*>(compressed.data());
        auto compressed_ptr_double = reinterpret_cast<double*>(compressed_ptr_size_t+2);
        auto compressed_ptr = reinterpret_cast<unsigned char*>(compressed_ptr_double+1);
        compressed_ptr_size_t[0] = logs_comp.size_in_bytes();
        compressed_ptr_size_t[1] = 0;
        compressed_ptr_double[0] = threshold;
        memcpy(compressed_ptr, logs_comp.data(), logs_comp.size_in_bytes());
      }

      return compressed;
    }
    double pw_rel;
    pressio_data const& input;
    pressio_compressor& abs_comp;
    pressio_compressor& sign_comp;
  };

  struct decompressor {

    template <class T>
    void restore_no_sign(T* begin, size_t n, T threshold) {
      for (size_t i = 0; i < n; ++i) {
        if(begin[i] < threshold) begin[i] = 0;
        else begin[i] = exp2(begin[i]);
      }
    }

    template <class T>
    void restore_signed(T* begin, bool* signs, size_t n, T threshold) {
      for (size_t i = 0; i < n; ++i) {
        if(begin[i] < threshold) begin[i] = 0;
        else begin[i] = exp2(begin[i]);

        if(signs[i]) begin[i] = -begin[i];
      }
    }

    void operator()(unsigned char* begin) {
      const size_t* sizes = reinterpret_cast<size_t*>(begin);
      const size_t logs_size = sizes[0];
      const size_t signs_size = sizes[1];
      const double threshold = *reinterpret_cast<double*>(begin + 2*sizeof(size_t));
      pressio_data logs_in = pressio_data::nonowning(pressio_byte_dtype, begin+2*sizeof(size_t)+sizeof(double), {logs_size});

      if(abs_comp->decompress(&logs_in, output) > 0) {
        throw std::runtime_error(abs_comp->error_msg());
      }

      if(signs_size) {
        //we need to restore signs
        pressio_data signs_in = pressio_data::nonowning(pressio_byte_dtype, begin+2*sizeof(size_t)+sizeof(double)+logs_size, {signs_size});
        pressio_data signs_out = pressio_data::owning(pressio_bool_dtype, output->dimensions());
        if(sign_comp->decompress(&signs_in, &signs_out) > 0) {
          throw std::runtime_error(sign_comp->error_msg());
        }
        switch(output->dtype()) {
          case pressio_float_dtype:
            restore_signed(static_cast<float*>(output->data()), static_cast<bool*>(signs_out.data()),output->num_elements(), static_cast<float>(threshold));
            return;
          case pressio_double_dtype:
            restore_signed(static_cast<double*>(output->data()), static_cast<bool*>(signs_out.data()),output->num_elements(), threshold);
            return;
          default:
            throw std::runtime_error("unsupported type");
        }
      } else {
        switch(output->dtype()) {
          case pressio_float_dtype:
            restore_no_sign(static_cast<float*>(output->data()), output->num_elements(), static_cast<float>(threshold));
            return;
          case pressio_double_dtype:
            restore_no_sign(static_cast<double*>(output->data()),output->num_elements(), threshold);
            return;
          default:
            throw std::runtime_error("unsupported type");
        }
      }
    }
    pressio_data* output;
    pressio_compressor& abs_comp;
    pressio_compressor& sign_comp;
  };

class pw_rel_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "pw_rel:abs_comp", abs_comp_id, abs_comp);
    set_meta(options, "pw_rel:sign_comp", signs_comp_id, signs_comp);
    set(options, "pressio:pw_rel", pw_rel);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "pw_rel:abs_comp", compressor_plugins(), abs_comp);
    set_meta_configuration(options, "pw_rel:sign_comp", compressor_plugins(), signs_comp);
    set(options, "pressio:thread_safe", std::min(get_threadsafe(*abs_comp), get_threadsafe(*signs_comp)));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "pw_rel:abs_comp", "compressor that supports an absolute error bound", abs_comp);
    set_meta_docs(options, "pw_rel:sign_comp", "compressor that compresses signs", signs_comp);
    set(options, "pressio:description", R"(abstraction for adapting an absolute error bound to a pw_rel error bound

    Adapted for LibPressio by Robert Underwood
    Algorithm by
    X. Liang, S. Di, D. Tao, Z. Chen, and F. Cappello, “An Efficient Transformation Scheme for Lossy Data Compression with Point-Wise Relative Error Bound,” in 2018 IEEE International Conference on Cluster Computing (CLUSTER), Sep. 2018, pp. 179–189. doi: 10.1109/CLUSTER.2018.00036.
    )");
    set(options, "pressio:pw_rel", R"(point wise relative error bound mode)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "pw_rel:abs_comp", compressor_plugins(), abs_comp_id, abs_comp);
    get_meta(options, "pw_rel:sign_comp", compressor_plugins(), signs_comp_id, signs_comp);
    get(options, "pressio:pw_rel", &pw_rel);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      switch(input->dtype()){
        case pressio_float_dtype:
          *output = compressor{pw_rel, *input, abs_comp, signs_comp}(static_cast<float*>(input->data()), static_cast<float*>(input->data()) + input->num_elements());
          break;
        case pressio_double_dtype:
          *output = compressor{pw_rel, *input, abs_comp, signs_comp}(static_cast<double*>(input->data()), static_cast<double*>(input->data()) + input->num_elements());
          break;
        default:
          return set_error(2, "unsupported type");
      }
      return 0;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try {
      decompressor{output, abs_comp, signs_comp}(static_cast<unsigned char*>(input->data()));
      return 0;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "pw_rel"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts;
    opts.copy_from(abs_comp->get_metrics_results());
    opts.copy_from(signs_comp->get_metrics_results());
    return opts;
  }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
      abs_comp->set_name(new_name + "/logs");
      signs_comp->set_name(new_name + "/signs");
    } else {
      abs_comp->set_name(new_name);
      signs_comp->set_name(new_name);
    }
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<pw_rel_compressor_plugin>(*this);
  }

  pressio_compressor abs_comp = compressor_plugins().build("noop");
  pressio_compressor signs_comp = compressor_plugins().build("noop");
  std::string abs_comp_id = "noop", signs_comp_id = "noop";
  double pw_rel = 1e-3;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "pw_rel", []() {
  return compat::make_unique<pw_rel_compressor_plugin>();
});

} }
