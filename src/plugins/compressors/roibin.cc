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
    set(options, "roibin:rows", peak_rows);
    set(options, "roibin:cols", peak_cols);
    set(options, "roibin:segs", peak_segs);
    set(options, "roibin:roi_size", roi_size);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "experimental");
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:description", R"(ROIBIN metacompressor
    
    This module treats compression as two tasks: region of interest saving + background saving
    Each is then forwarded on to a separate compressor for additional compression.
    )");
    set_meta_docs(options, "roibin:roi", "region of interest compression", roi);
    set_meta_docs(options, "roibin:background", "background compression", background);
    set(options, "roibin:rows", "row locations");
    set(options, "roibin:cols", "column locations");
    set(options, "roibin:segs", "segment locations");
    set(options, "roibin:roi_size", "region of interest size");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "roibin:roi", compressor_plugins(), roi_id, roi);
    get_meta(options, "roibin:background", compressor_plugins(), background_id, background);
    get(options, "roibin:rows", &peak_rows);
    get(options, "roibin:cols", &peak_cols);
    get(options, "roibin:segs", &peak_segs);
    get(options, "roibin:roi_size", &roi_size);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    try {
      pressio_data roi_data = save_roi(*input);
      pressio_data roi_compressed = pressio_data::empty(pressio_byte_dtype, {});
      pressio_data background_compressed = pressio_data::empty(pressio_byte_dtype, {});
      int ec = roi->compress(&roi_data, &roi_compressed);
      if(ec < 0) {
        set_error(ec, roi->error_msg());
      } else if (ec > 0) {
        return set_error(ec, roi->error_msg());
      }

      ec = background->compress(input, &background_compressed);
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
    size_t regions = std::min(peak_rows.get_dimension(0), std::min(peak_cols.get_dimension(0), peak_segs.get_dimension(0)));
    pressio_data roi_decompressed = pressio_data::owning(output->dtype(), roi_dims(output->get_dimension(3) * regions));
    
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

    return 0;

  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "roibin"; }

  void set_name_impl(std::string const& new_name) override {
    roi->set_name(new_name + "/roi");
    background->set_name(new_name + "/background");
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
    auto roi_size = this->roi_size.to_vector<size_t>();
    if(roi_size.at(3) != 0 || roi_size.at(2) != 0 || roi_size.at(1) == 0 || roi_size.at(0) == 0) {
        throw std::runtime_error("unsupported roi_size");
    }
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

  std::vector<std::array<size_t, 4>> centers(size_t events) const {
    std::vector<std::array<size_t, 4>> centers;
    auto rows = peak_rows.to_vector<size_t>();
    auto cols = peak_cols.to_vector<size_t>();
    auto segs = peak_segs.to_vector<size_t>();
    size_t regions = std::min(rows.size(), std::min(cols.size(), segs.size()));

    centers.resize(regions * events);
    for (size_t i = 0; i < regions; ++i) {
      for (size_t j = 0; j < events; ++j) {
        centers[i*events+j][0] = rows[i];
        centers[i*events+j][1] = cols[i];
        centers[i*events+j][2] = segs[i];
        centers[i*events+j][3] = j;
      }
    }
    return centers;
  }

  template <class T>
  pressio_data save_roi_typed(pressio_data const& data) {
    indexer<4> id{
      data.get_dimension(0),
      data.get_dimension(1),
      data.get_dimension(2),
      data.get_dimension(3)
    };

    auto centers = this->centers(data.get_dimension(3));
    auto roi_size_v = this->roi_size.to_vector<size_t>();
    indexer<4> roi_size{roi_size_v.begin(), roi_size_v.end()};
    indexer<5> roi = to_roimem(roi_size, centers.size());
    pressio_data roi_mem = pressio_data::owning(data.dtype(), roi.as_vec());
    roi_save(id, roi_size, roi, centers, static_cast<T const*>(data.data()), static_cast<T*>(roi_mem.data()));
    return roi_mem;
  }

  void restore_roi(pressio_data& data, pressio_data const& roi) {
    auto roi_size = this->roi_size.to_vector<size_t>();
    if(roi_size.at(3) != 0 || roi_size.at(2) != 0 || roi_size.at(1) == 0 || roi_size.at(0) == 0) {
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
    indexer<4> id{
      restored.get_dimension(0),
      restored.get_dimension(1),
      restored.get_dimension(2),
      restored.get_dimension(3)
    };

    auto roi_size_v = this->roi_size.to_vector<size_t>();
    auto centers = this->centers(restored.get_dimension(3));
    indexer<4> roi_size{roi_size_v.begin(), roi_size_v.end()};
    indexer<5> roi = to_roimem(roi_size, centers.size());
    roi_restore(id, roi_size, roi, centers, static_cast<T const*>(roi_mem.data()), static_cast<T*>(restored.data()));
  }

  std::vector<size_t> roi_dims(size_t events) {
    auto roi_size_v = this->roi_size.to_vector<size_t>();
    indexer<4> roi_size{roi_size_v.begin(), roi_size_v.end()};
    indexer<5> roi = to_roimem(roi_size, events);
    std::vector<size_t> v(5);
    for (int i = 0; i < 5; ++i) {
      v[i] = roi[i];
    }
    return v;
  }

  std::string background_id = "noop";
  std::string roi_id = "noop";
  pressio_compressor background = compressor_plugins().build("noop");
  pressio_compressor roi = compressor_plugins().build("noop");
  pressio_data peak_rows, peak_segs, peak_cols, roi_size{5,5,0,0};
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "roibin", []() {
  return compat::make_unique<roibin_compressor_plugin>();
});

} }
