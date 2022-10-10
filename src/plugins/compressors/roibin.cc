#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "roibin_impl.h"

namespace libpressio { namespace roibin_ns {

class roibin_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "roibin:roi", roi_id, roi);
    set_meta(options, "roibin:background", background_id, background);
    set(options, "roibin:centers", centers);
    set(options, "roibin:roi_size", roi_size);
    set(options, "roibin:nthreads", n_threads);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    options.copy_from(background->get_configuration());
    options.copy_from(roi->get_configuration());
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "roibin:roi", "region of interest compression", roi);
    set_meta_docs(options, "roibin:background", "background compression", background);
    set(options, "pressio:description", R"(ROIBIN metacompressor
    
    This module treats compression as two tasks: region of interest saving + background saving
    Each is then forwarded on to a separate compressor for additional compression.
    )");
    set(options, "roibin:centers", "centers of the region of interest");
    set(options, "roibin:roi_size", "region of interest size");
    set(options, "roibin:nthreads", "number of threads for region of interest");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "roibin:roi", compressor_plugins(), roi_id, roi);
    get_meta(options, "roibin:background", compressor_plugins(), background_id, background);
    get(options, "roibin:centers", &centers);
    get(options, "roibin:roi_size", &roi_size);
    uint32_t tmp;
    if(get(options, "roibin:nthreads", &tmp) == pressio_options_key_set) {
      if(tmp > 0) {
        n_threads = tmp;
      }
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      pressio_data roi_data = save_roi(*input);
      pressio_data roi_compressed = pressio_data::empty(pressio_byte_dtype, {});
      pressio_data background_compressed = pressio_data::empty(pressio_byte_dtype, {});
      if(roi_data.size_in_bytes() > 0) {
        int ec = roi->compress(&roi_data, &roi_compressed);
        if(ec < 0) {
          set_error(ec, roi->error_msg());
        } else if (ec > 0) {
          return set_error(ec, roi->error_msg());
        }
      }

      int ec = background->compress(input, &background_compressed);
      if(ec < 0) {
        set_error(ec, background->error_msg());
      } else if (ec > 0) {
        return set_error(ec, background->error_msg());
      }

      *output = pressio_data::owning(
          pressio_byte_dtype,
          {(roi_compressed.size_in_bytes() + background_compressed.size_in_bytes() + 2*sizeof(size_t))}
          );
      size_t* sizes = static_cast<size_t*>(output->data());
      sizes[0] = roi_compressed.size_in_bytes();
      sizes[1] = background_compressed.size_in_bytes();
      memcpy(static_cast<uint8_t*>(output->data()) + 2*sizeof(size_t), roi_compressed.data(), roi_compressed.size_in_bytes());
      memcpy(static_cast<uint8_t*>(output->data()) + roi_compressed.size_in_bytes() + 2*sizeof(size_t), background_compressed.data(), background_compressed.size_in_bytes());
      return 0;
    } catch(std::exception const& ex) {
      return set_error(1, ex.what());
    }
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    const size_t* sizes = static_cast<size_t*>(input->data());
    pressio_data roi_compressed = pressio_data::nonowning(pressio_byte_dtype, 
        static_cast<uint8_t*>(input->data()) + 2*sizeof(size_t),
        {sizes[0]}); 
    pressio_data backround_compressed = pressio_data::nonowning(pressio_byte_dtype, 
        static_cast<uint8_t*>(input->data()) + sizes[0] + 2*sizeof(size_t),
        {sizes[1]}); 
    if(roi_compressed.size_in_bytes() > 0) {
      size_t regions = centers.get_dimension(1);
      pressio_data roi_decompressed = pressio_data::owning(output->dtype(), roi_dims(regions));
      
      int ec = roi->decompress(&roi_compressed, &roi_decompressed);
      if(ec < 0) {
        set_error(ec, roi->error_msg());
      } else if (ec > 0) {
        return set_error(ec, roi->error_msg());
      }
      ec = background->decompress(&backround_compressed, output);
      if(ec < 0) {
        set_error(ec, background->error_msg());
      } else if (ec > 0) {
        return set_error(ec, background->error_msg());
      }

      try {
        restore_roi(*output, roi_decompressed);
      } catch (std::exception const& ex) {
        return set_error(2, ex.what());
      }
    } else {
      int ec = background->decompress(&backround_compressed, output);
      if(ec < 0) {
        set_error(ec, background->error_msg());
      } else if (ec > 0) {
        return set_error(ec, background->error_msg());
      }
    }


    return 0;

  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "roibin"; }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
      roi->set_name(new_name + "/roi");
      background->set_name(new_name + "/background");
    } else {
      roi->set_name(new_name);
      background->set_name(new_name);
    }
  }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts = background->get_metrics_results();
    opts.copy_from(roi->get_metrics_results());
    return opts;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<roibin_compressor_plugin>(*this);
  }

  pressio_data save_roi(pressio_data const& data) {
    switch(data.dtype()) {
      case pressio_bool_dtype:
        return save_roi_typed<bool>(data);
      case pressio_int8_dtype:
        return save_roi_typed<int8_t>(data);
      case pressio_int16_dtype:
        return save_roi_typed<int16_t>(data);
      case pressio_int32_dtype:
        return save_roi_typed<int32_t>(data);
      case pressio_int64_dtype:
        return save_roi_typed<int64_t>(data);
      case pressio_uint8_dtype:
        return save_roi_typed<uint8_t>(data);
      case pressio_uint16_dtype:
        return save_roi_typed<uint16_t>(data);
      case pressio_uint32_dtype:
        return save_roi_typed<uint32_t>(data);
      case pressio_uint64_dtype:
        return save_roi_typed<uint64_t>(data);
      case pressio_double_dtype:
        return save_roi_typed<double>(data);
      case pressio_float_dtype:
        return save_roi_typed<float>(data);
      default:
        throw std::runtime_error("unsupported type");
    }
    
  }


  template <class T>
  pressio_data save_roi_typed(pressio_data const& data) {
    auto roi_size_v = this->roi_size.to_vector<size_t>();
    if(data.num_dimensions() != roi_size_v.size()) {
      throw std::runtime_error("roi_size must match data size");
    }
    if(data.num_dimensions() != centers.get_dimension(0) && centers.num_elements() > 0) {
      throw std::runtime_error("centers dimension must match num_dimensions of input data or be empty");
    }
    switch(data.num_dimensions()) {
      case 1:
        {
          indexer<1> id{data.get_dimension(0)};
          indexer<1> roi_size(roi_size_v.begin(), roi_size_v.end());
          indexer<2> roi_mem = to_roimem(roi_size, centers.get_dimension(1));
          pressio_data roi_mem_data = pressio_data::owning(data.dtype(), roi_mem.as_vec());
          roi_save(id, roi_size, roi_mem, centers, static_cast<T const*>(data.data()), static_cast<T*>(roi_mem_data.data()), n_threads);
          return roi_mem_data;
        }
      case 2:
        {
          indexer<2> id{data.get_dimension(0), data.get_dimension(1)};
          indexer<2> roi_size(roi_size_v.begin(), roi_size_v.end());
          indexer<3> roi_mem = to_roimem(roi_size, centers.get_dimension(1));
          pressio_data roi_mem_data = pressio_data::owning(data.dtype(), roi_mem.as_vec());
          roi_save(id, roi_size, roi_mem, centers, static_cast<T const*>(data.data()), static_cast<T*>(roi_mem_data.data()), n_threads);
          return roi_mem_data;
        }
      case 3:
        {
          indexer<3> id{data.get_dimension(0), data.get_dimension(1), data.get_dimension(2)};
          indexer<3> roi_size(roi_size_v.begin(), roi_size_v.end());
          indexer<4> roi_mem = to_roimem(roi_size, centers.get_dimension(1));
          pressio_data roi_mem_data = pressio_data::owning(data.dtype(), roi_mem.as_vec());
          roi_save(id, roi_size, roi_mem, centers, static_cast<T const*>(data.data()), static_cast<T*>(roi_mem_data.data()), n_threads);
          return roi_mem_data;
        }
      case 4:
        {
          indexer<4> id{data.get_dimension(0), data.get_dimension(1), data.get_dimension(2), data.get_dimension(3)};
          indexer<4> roi_size(roi_size_v.begin(), roi_size_v.end());
          indexer<5> roi_mem = to_roimem(roi_size, centers.get_dimension(1));
          pressio_data roi_mem_data = pressio_data::owning(data.dtype(), roi_mem.as_vec());
          roi_save(id, roi_size, roi_mem, centers, static_cast<T const*>(data.data()), static_cast<T*>(roi_mem_data.data()), n_threads);
          return roi_mem_data;
        }
      default:
        throw std::runtime_error("unsupported dimension " + std::to_string(data.num_dimensions()));
    }
  }

  void restore_roi(pressio_data& data, pressio_data const& roi) {
    auto roi_size = this->roi_size.to_vector<size_t>();
    if(roi_size.size() != data.num_dimensions()) {
        throw std::runtime_error("unsupported roi_size");
    }
    switch(data.dtype()) {
      case pressio_int8_dtype:
        restore_roi_typed<int8_t>(data, roi);
        return;
      case pressio_int16_dtype:
        restore_roi_typed<int16_t>(data, roi);
        return;
      case pressio_int32_dtype:
        restore_roi_typed<int32_t>(data, roi);
        return;
      case pressio_int64_dtype:
        restore_roi_typed<int64_t>(data, roi);
        return;
      case pressio_uint8_dtype:
        restore_roi_typed<uint8_t>(data, roi);
        return;
      case pressio_uint16_dtype:
        restore_roi_typed<uint16_t>(data, roi);
        return;
      case pressio_uint32_dtype:
        restore_roi_typed<uint32_t>(data, roi);
        return;
      case pressio_uint64_dtype:
        restore_roi_typed<uint64_t>(data, roi);
        return;
      case pressio_double_dtype:
        restore_roi_typed<double>(data, roi);
        return;
      case pressio_float_dtype:
        restore_roi_typed<float>(data, roi);
        return;
      default:
        throw std::runtime_error("unsupported type");
    }
  }

  template <class T>
  void restore_roi_typed(pressio_data & restored, pressio_data const& roi_mem) {
    switch(restored.num_dimensions()){
      case 1:
        restore_roi_sized<T,1>(restored, roi_mem);
        break;
      case 2:
        restore_roi_sized<T,2>(restored, roi_mem);
        break;
      case 3:
        restore_roi_sized<T,3>(restored, roi_mem);
        break;
      case 4:
        restore_roi_sized<T,4>(restored, roi_mem);
        break;
      default:
        throw std::runtime_error("unsupported number of dimensions " + std::to_string(restored.num_dimensions()));
    }
  }

  template <class T, size_t N>
  void restore_roi_sized(pressio_data& restored, pressio_data const& roi_mem) {
    auto restored_v = restored.dimensions();
    indexer<N> id(restored_v.begin(), restored_v.end());

    auto roi_size_v = this->roi_size.to_vector<size_t>();
    indexer<N> roi_size{roi_size_v.begin(), roi_size_v.end()};
    indexer<N+1> roi = to_roimem(roi_size, centers.get_dimension(1));
    roi_restore(id, roi_size, roi, centers, static_cast<T const*>(roi_mem.data()), static_cast<T*>(restored.data()), n_threads);

  }

  std::vector<size_t> roi_dims(size_t num_centers) {
    auto roi_size_v = this->roi_size.to_vector<size_t>();
    std::vector<size_t> dims(roi_size_v.size() + 1);
    for (size_t i = 0; i < roi_size_v.size(); ++i) {
      dims[i] = roi_size_v[i]*2 +1;
    }
    dims.back() = num_centers;
    return dims;
  }

  std::string background_id = "noop";
  std::string roi_id = "noop";
  pressio_compressor background = compressor_plugins().build("noop");
  pressio_compressor roi = compressor_plugins().build("noop");
  pressio_data roi_size{5,5,0,0};
  pressio_data centers;
  uint32_t n_threads = 1;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "roibin", []() {
  return compat::make_unique<roibin_compressor_plugin>();
});

} }
