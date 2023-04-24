#include "std_compat/cmath.h"
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

#include "basic_indexer.h"

namespace libpressio { namespace mask_interpolation_ns {

    using namespace utilities;

class mask_interpolation_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "mask_interpolation:compressor", comp_id, comp);
    set(options, "mask_interpolation:mask", mask);
    set(options, "mask_interpolation:fill", fill);
    set(options, "mask_interpolation:mask_mode", mask_mode);
    set(options, "pressio:nthreads", nthreads);
    set(options, "mask_interpolation:nthreads", nthreads);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "mask_interpolation:compressor", compressor_plugins(), comp);
    set(options, "pressio:thread_safe", get_threadsafe(*comp));
    set(options, "pressio:stability", "experimental");
    set(options, "mask_interpolation:mask_mode", std::vector<std::string>{"fill", "interp"});
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "mask_interpolation:compressor", "compressor to use after masking", comp);
    set(options, "pressio:description", R"(apply interpolation to replace masked values)");
    set(options, "mask_interpolation:mask", "mask of values to replace true means replace");
    set(options, "mask_interpolation:fill", "for values that cannot be interpolated, what value to fill with");
    set(options, "mask_interpolation:mask_mode", "type of interpolation to use for masked values");
    set(options, "mask_interpolation:nthreads", "number of execution threads");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "mask_interpolation:compressor", compressor_plugins(), comp_id, comp);
    uint32_t tmp_nthreads;
    if(get(options, "pressio:nthreads", &tmp_nthreads) == pressio_options_key_set) {
        if(tmp_nthreads > 0) nthreads = tmp_nthreads;
    }
    if(get(options, "mask_interpolation:nthreads", &tmp_nthreads) == pressio_options_key_set){
        if(tmp_nthreads > 0) nthreads = tmp_nthreads;
    }
    get(options, "mask_interpolation:mask", &mask);
    get(options, "mask_interpolation:fill", &fill);
    std::string tmp;
    if(get(options, "mask_interpolation:mask_mode", &tmp)==pressio_options_key_set) {
        if(tmp == "fill" || tmp == "interp") {
            mask_mode = std::move(tmp);
        } else {
            set_error(1, "invalid fill mode");
        }
    }

    return 0;
  }

  struct interp_apply_mask {
      template <class T, class V>
      int operator()(T* input_ptr, T* input_ptr_end, V* output_ptr, V* output_ptr_end) {
          std::vector<size_t> dims = data.dimensions();
          if(impl->mask.dtype() != pressio_uint8_dtype) {
              impl->set_error(2, "mask must be uint8_t");
              return 2;
          }
          uint8_t* mask_ptr = static_cast<uint8_t*>(impl->mask.data());
          T fill = static_cast<T>(impl->fill);
          if(dims.size() == 1) {
              indexer<1> idx{dims.at(0)};
              for(size_t i = 0; i < idx[0]; ++i) {
                  if(!mask_ptr[idx(i)]) {
                      output_ptr[idx(i)] = input_ptr[idx(i)];
                  } else {
                      T left = fill;
                      size_t left_pos = 0;
                      for (size_t p = 1; p <= i; ++p) {
                          size_t new_idx = idx(i-p);
                          if(mask_ptr[new_idx]) {
                              left_pos = p;
                              left = input_ptr[new_idx];
                              break;
                          }
                      }
                      T right = fill;
                      size_t right_pos = 0;
                      for (size_t p = 1; p+i < idx.max_dims[0]; ++p) {
                          size_t new_idx = idx(i+p);
                          if(mask_ptr[new_idx]) {
                              right_pos = p;
                              right = input_ptr[new_idx];
                              break;
                          }
                      }

                      if(left_pos == 0 && right_pos == 0) {
                          output_ptr[idx(i)] = fill;
                      } else if (left_pos == 0) {
                          output_ptr[idx(i)] = right;
                      } else if (right_pos == 0) {
                          output_ptr[idx(i)] = left;
                      } else {
                          return compat::lerp(left, right, static_cast<double>(left_pos)/static_cast<double>(left_pos+right_pos));
                      }
                  }
              }
          } else if(dims.size() == 2) {
              indexer<2> idx{dims.at(0), dims.at(1)};
#pragma omp parallel for num_threads(impl->nthreads)
              for(size_t j = 0; j < idx[1]; ++j) {
              for(size_t i = 0; i < idx[0]; ++i) {
                  if(!mask_ptr[idx(i,j)]) {
                      output_ptr[idx(i,j)] = input_ptr[idx(i,j)];
                  } else {
                      T left = fill;
                      size_t left_pos = 0;
                      for (size_t p = 1; p <= i; ++p) {
                          size_t new_idx = idx(i-p,j);
                          if(!mask_ptr[new_idx]) {
                              left_pos = p;
                              left = input_ptr[new_idx];
                              break;
                          }
                      }
                      T right = fill;
                      size_t right_pos = 0;
                      for (size_t p = 1; p+i < idx.max_dims[0]; ++p) {
                          size_t new_idx = idx(i+p,j);
                          if(!mask_ptr[new_idx]) {
                              right_pos = p;
                              right = input_ptr[new_idx];
                              break;
                          }
                      }

                      T top = fill;
                      size_t top_pos = 0;
                      for (size_t p = 1; p <= j; ++p) {
                          size_t new_idx = idx(i,j-p);
                          if(!mask_ptr[new_idx]) {
                              top_pos = p;
                              top = input_ptr[new_idx];
                              break;
                          }
                      }
                      T bot = fill;
                      size_t bot_pos = 0;
                      for (size_t p = 1; p+j < idx.max_dims[1]; ++p) {
                          size_t new_idx = idx(i,j+p);
                          if(!mask_ptr[new_idx]) {
                              bot_pos = p;
                              bot = input_ptr[new_idx];
                              break;
                          }
                      }

                      if(left_pos) {
                          if(right_pos) {
                              auto left_right = compat::lerp(left, right, static_cast<double>(left_pos)/static_cast<double>(left_pos+right_pos));
                              if(top_pos) {
                                  if(bot_pos) {
                                      //left, right, top, bot -> interp lrtb
                                      auto top_bot = compat::lerp(top, bot, static_cast<double>(top_pos)/static_cast<double>(top_pos+bot_pos));
                                      output_ptr[idx(i,j)] = compat::lerp(left_right, top_bot, static_cast<double>(top_pos+bot_pos)/static_cast<double>(left_pos+right_pos+top_pos+bot_pos));
                                  } else {
                                      //left, right, top, ~bot -> interp lr
                                      output_ptr[idx(i,j)] = left_right;
                                  }
                              } else {
                                  //left, right, ~top, bot -> interp lr
                                  //left, right, ~top, ~bot -> interp lr
                                  output_ptr[idx(i,j)] = left_right;
                              }
                          } else {
                              if(top_pos) {
                                  if(bot_pos) {
                                      //left, ~right, top, bot -> interp lt
                                      auto top_bot = compat::lerp(top, bot, static_cast<double>(top_pos)/static_cast<double>(top_pos+bot_pos));
                                      output_ptr[idx(i,j)] = top_bot;
                                  } else {
                                      //left, ~right, top, ~bot -> avg lt
                                      output_ptr[idx(i,j)] =
                                          left * static_cast<double>(top_pos)/static_cast<double>(top_pos+left_pos) +
                                          top * static_cast<double>(left_pos)/static_cast<double>(top_pos+left_pos);
                                  }
                              } else {
                                  if(bot_pos) {
                                      //left, ~right, ~top, bot -> avg lb
                                      output_ptr[idx(i,j)] =
                                          left * static_cast<double>(bot_pos)/static_cast<double>(bot_pos+left_pos) +
                                          bot * static_cast<double>(left_pos)/static_cast<double>(bot_pos+left_pos);
                                  } else {
                                      //left, ~right, ~top, ~bot -> left
                                      output_ptr[idx(i,j)] = left;
                                  }
                              }
                          }
                      } else {
                          if(right_pos) {
                              if(top_pos) {
                                  if(bot_pos) {
                                      auto top_bot = compat::lerp(top, bot, static_cast<double>(top_pos)/static_cast<double>(top_pos+bot_pos));
                                      //~left, right, top, bot
                                      output_ptr[idx(i,j)] = top_bot;
                                  } else {
                                      //~left, right, top, ~bot
                                      output_ptr[idx(i,j)] =
                                          right * static_cast<double>(top_pos)/static_cast<double>(top_pos+right_pos) +
                                          top * static_cast<double>(right_pos)/static_cast<double>(top_pos+right_pos);
                                  }
                              } else {
                                  if(bot_pos) {
                                      //~left, right, ~top, bot
                                      output_ptr[idx(i,j)] =
                                          right * static_cast<double>(bot_pos)/static_cast<double>(bot_pos+right_pos) +
                                          bot * static_cast<double>(right_pos)/static_cast<double>(bot_pos+right_pos);
                                  } else {
                                      //~left, right, ~top, ~bot -> right
                                      output_ptr[idx(i,j)] = right;
                                  }
                              }
                          } else {
                              if(top_pos) {
                                  if(bot_pos) {
                                      auto top_bot = compat::lerp(top, bot, static_cast<double>(top_pos)/static_cast<double>(top_pos+bot_pos));
                                      //~left, ~right, top, bot -> interp tb
                                      output_ptr[idx(i,j)] = top_bot;
                                  } else {
                                      //~left, ~right, top, ~bot -> top
                                      output_ptr[idx(i,j)] = top;
                                  }
                              } else {
                                  if(bot_pos) {
                                      //~left, ~right, ~top, bot -> bot
                                      output_ptr[idx(i,j)] = bot;
                                  } else {
                                      //~left, ~right, ~top, ~bot -> fill
                                      output_ptr[idx(i,j)] = fill;
                                  }
                              }
                          }
                      }
                  }
              }}
          } else {
              auto norm_data = pressio_data::nonowning(data.dtype(), input_ptr, data.normalized_dims(compat::nullopt));
              if(norm_data.num_dimensions() <= 2) {
                  return interp_apply_mask{impl, norm_data}(input_ptr, input_ptr_end, output_ptr, output_ptr_end);
              } else {
                  return impl->set_error(1, "not supported: data is too high dimension for mask_interp");
              }
          }

          return 0;
      }
      mask_interpolation_compressor_plugin* impl;
      pressio_data const& data;
  };
  struct fill_apply_mask {
      template <class T, class V>
      int operator()(T* input_ptr, T*, V* output_ptr, V*) {
          (void)input_ptr;
          (void)output_ptr;
          if(impl->mask.dtype() != pressio_uint8_dtype) {
              impl->set_error(2, "mask must be uint8_t");
              return 2;
          }
          uint8_t* mask_ptr = static_cast<uint8_t*>(impl->mask.data());

          size_t total_len = data.num_elements();
#pragma omp parallel for num_threads(impl->nthreads)
          for (size_t i = 0; i < total_len; ++i) {
              if(mask_ptr[i]) {
                  output_ptr[i] = static_cast<T>(impl->fill);
              } else {
                  output_ptr[i] = input_ptr[i];
              }
          }

          return 0;
      }
      mask_interpolation_compressor_plugin* impl;
      pressio_data const& data;
  };


  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
     pressio_data masked = pressio_data::clone(*input);

     int rc = 0;
     if(mask_mode == "fill") {
         rc = pressio_data_for_each<int>(*input, masked, fill_apply_mask{this, *input});
     } else if(mask_mode == "interp") {
         rc = pressio_data_for_each<int>(*input, masked, interp_apply_mask{this, *input});
     }
     if(rc) {
         return rc;
     }

     rc = comp->compress(&masked, output);
     if(rc) set_error(rc, comp->error_msg());
     return rc;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
     int rc = comp->decompress(input, output);
     if(rc) set_error(rc, comp->error_msg());
     return rc;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "mask_interpolation"; }

  void set_name_impl(std::string const& new_name) override {
      comp->set_name(new_name + "/" + comp->prefix());
  }
  std::vector<std::string> children_impl() const final {
      return { comp->get_name() };
  }

  pressio_options get_metrics_results_impl() const override {
    return comp->get_metrics_results();
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<mask_interpolation_compressor_plugin>(*this);
  }

  double fill = 0.0;
  pressio_data mask;
  std::string comp_id = "noop";
  pressio_compressor comp = compressor_plugins().build(comp_id);
  std::string mask_mode = "fill";
  uint32_t nthreads = 1;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "mask_interpolation", []() {
  return compat::make_unique<mask_interpolation_compressor_plugin>();
});

} }
