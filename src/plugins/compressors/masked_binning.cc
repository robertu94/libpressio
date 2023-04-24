#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "roibin_impl.h"

namespace libpressio { namespace mask_binning_ns {

    using namespace utilities;

  struct bin_op {
    template<class T>
    pressio_data operator()(T const* t, T const*) {
      auto bins_v = this->bins.to_vector<size_t>();
      if(bins_v.size() != dims.size()) {
        throw std::runtime_error("dims size does not match bins size");
      }
      auto mask_ptr = static_cast<const bool*>(mask.data());

      switch(dims.size()) {
        case 1:
          {
          indexer<1> id{dims.at(0)};
          indexer<1> bins{bins_v.begin(), bins_v.end()};
          indexer<1> binned_storage = roibin_ns::to_binned_index(id, bins);
          pressio_data binned = pressio_data::owning( pressio_dtype_from_type<T>(), {binned_storage[0]});
          roibin_ns::mask_bin_omp(id, binned_storage, bins, static_cast<T const*>(t), static_cast<T*>(binned.data()), mask_ptr, n_threads);
          return binned;
          }
        case 2:
          {
          indexer<2> id{dims.at(0), dims.at(1)};
          indexer<2> bins{bins_v.begin(), bins_v.end()};
          indexer<2> binned_storage = roibin_ns::to_binned_index(id, bins);
          pressio_data binned = pressio_data::owning( pressio_dtype_from_type<T>(), {binned_storage[0], binned_storage[1]});
          roibin_ns::mask_bin_omp(id, binned_storage, bins, static_cast<T const*>(t), static_cast<T*>(binned.data()), mask_ptr, n_threads);
          return binned;
          }
        case 3:
          {
          indexer<3> id{dims.at(0), dims.at(1), dims.at(2)};
          indexer<3> bins{bins_v.begin(), bins_v.end()};
          indexer<3> binned_storage = roibin_ns::to_binned_index(id, bins);
          pressio_data binned = pressio_data::owning(
              pressio_dtype_from_type<T>(), {binned_storage[0], binned_storage[1], binned_storage[2]});
          roibin_ns::mask_bin_omp(id, binned_storage, bins, static_cast<T const*>(t), static_cast<T*>(binned.data()), mask_ptr, n_threads);
          return binned;
          }
        case 4:
          {
          indexer<4> id{ dims.at(0), dims.at(1), dims.at(2), dims.at(3) };
          indexer<4> bins{bins_v.begin(), bins_v.end()};
          indexer<4> binned_storage = roibin_ns::to_binned_index(id, bins);
          pressio_data binned = pressio_data::owning( pressio_dtype_from_type<T>(),
              {binned_storage[0], binned_storage[1], binned_storage[2], binned_storage[3]});
          roibin_ns::mask_bin_omp(id, binned_storage, bins, static_cast<T const*>(t), static_cast<T*>(binned.data()), mask_ptr, n_threads);
          return binned;
          }
        default:
          throw std::runtime_error("unsupported mask_binning dimension " + std::to_string(dims.size()));
      }

    }

    std::vector<size_t> const& dims;
    pressio_data const& bins;
    pressio_data const& mask;
    uint32_t n_threads;
  };

  template <size_t N>
  struct restore_op {
    template <class T, class V>
    int operator()(T * input_data, T *, V* output_data) {
        roibin_ns::restore_omp(id, binned_storage, bins, input_data, reinterpret_cast<T*>(output_data), n_threads);
      return 0;
    }

    indexer<N> id;
    indexer<N> bins;
    indexer<N> binned_storage;
    uint32_t n_threads;
  };

class binning_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "mask_binning:compressor", comp_id, comp);
    set(options, "mask_binning:shape", bins);
    set(options, "pressio:nthreads", n_threads);
    set(options, "mask_binning:nthreads", n_threads);
    set(options, "mask_binning:mask", mask);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "mask_binning:compressor", compressor_plugins(), comp);
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "mask_binning:compressor", comp_id, comp);
    set(options, "pressio:description", R"(preforms a mask_binning operation on the input data on compression and extrapolates on decompression)");
    set(options, "mask_binning:shape", "shape of the bins to apply");
    set(options, "mask_binning:nthreads", "number of cpu threads to use for mask_binning");
    set(options, "mask_binning:mask", "boolean mask to apply, true means ignore");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "mask_binning:compressor", compressor_plugins(), comp_id, comp);
    get(options, "mask_binning:shape", &bins);
    {
      uint32_t tmp;
      if(get(options, "pressio:nthreads", &tmp) == pressio_options_key_set) {
        if(tmp > 0) {
          n_threads = tmp;
        } else {
          return set_error(1, "threads must be positive");
        }
      }
    }
    {
      uint32_t tmp;
      if(get(options, "mask_binning:nthreads", &tmp) == pressio_options_key_set) {
        if(tmp > 0) {
          n_threads = tmp;
        } else {
          return set_error(1, "threads must be positive");
        }
      }
    }
    if(get(options, "mask_binning:mask", &mask) == pressio_options_key_set) {
        if(mask.dtype() != pressio_bool_dtype) {
            mask = mask.cast(pressio_bool_dtype);
        }
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      auto tmp = pressio_data_for_each<pressio_data>(*input, bin_op{input->dimensions(), bins, mask, n_threads});
      int rc = comp->compress(&tmp, output);
      if(rc) {
        return set_error(comp->error_code(), comp->error_msg());
      } else {
        return 0;
      }
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

public:

  template <size_t N>
  int decompress_impl_sized(pressio_data const* input, pressio_data *output) {
      auto dims_v = output->dimensions();
      if (dims_v.size() != N) {
        throw std::runtime_error("mismatch in size in dims");
      }
      indexer<N> id(dims_v.begin(), dims_v.end());

      auto bins_v = this->bins.to_vector<size_t>();
      if (bins_v.size() != N) {
        throw std::runtime_error("mismatch in size in bins");
      }
      indexer<N> bins{bins_v.begin(), bins_v.end()};

      indexer<N> binned_storage = roibin_ns::to_binned_index(id, bins);
      pressio_data tmp_out = pressio_data::owning(output->dtype(), binned_storage.as_vec());
      int rc = comp->decompress(input, &tmp_out);

      if(rc > 0) {
        return set_error(comp->error_code(), comp->error_msg());
      } else if(rc < 0) {
        set_error(comp->error_code(), comp->error_msg());
      }

      pressio_data_for_each<int>(tmp_out, *output, restore_op<N>{id, bins, binned_storage, n_threads});
      return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    try {
      switch(output->num_dimensions()) {
        case 1:
          return decompress_impl_sized<1>(input,output);
        case 2:
          return decompress_impl_sized<2>(input,output);
        case 3:
          return decompress_impl_sized<3>(input,output);
        case 4:
          return decompress_impl_sized<4>(input,output);
        default:
          throw std::runtime_error("unsupported dimension " + std::to_string(output->num_dimensions()));
      }
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

public:
  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "mask_binning"; }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
    comp->set_name(new_name + "/" + comp->prefix()); 
    } else {
    comp->set_name(new_name); 
    }
  }
  std::vector<std::string> children_impl() const final {
      return { comp->get_name() };
  }

  pressio_options get_metrics_results_impl() const override {
    return comp->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<binning_compressor_plugin>(*this);
  }

  pressio_data bins{2,2,1,1};
  pressio_data mask = pressio_data::empty(pressio_bool_dtype, {});
  pressio_compressor comp = compressor_plugins().build("noop");
  std::string comp_id = "noop";
  uint32_t n_threads = 1;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "mask_binning", []() {
  return compat::make_unique<binning_compressor_plugin>();
});

} }
