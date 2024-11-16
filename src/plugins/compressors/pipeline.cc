
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "libpressio_ext/cpp/pressio.h"

namespace libpressio { namespace pipeline_ns {

class pipeline_compressor_plugin : public libpressio_compressor_plugin {
public:
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta_many(options, "pipeline:pipeline", ids, plugins);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_many_configuration(options, "pipeline:pipeline", compressor_plugins(), plugins);
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {}; 
    std::vector<pressio_configurable const*> invalidation_children {}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{}));
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_many_docs(options, "pipeline:pipeline", "series of plugins to invoke", plugins);
    set(options, "pipeline:names", "pipeline element names");
    set(options, "pressio:description", R"(invoke a series of compressors in a pipeline)");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get(options, "pipeline:names", &names);
    get_meta_many(options, "pipeline:pipeline", compressor_plugins(), ids, plugins);
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    int ec = 0;
    pressio_data tmp_in = pressio_data::nonowning(*input);
    pressio_data tmp_out = pressio_data::empty(pressio_byte_dtype, {});
    std::vector<pressio_data> metadata;

    /*
     *   |--------------------------------|
     *   | uint64_t format_id   =1        |
     *   | int64_t num_metadata          |
     *   |--------------------------------|
     *   | struct stage_dims {            |
     *   | uint64_t num_dim               |
     *   | uint64_t(pressio_dtype) type   |
     *   | uint64_t[num_dim] dim          |
     *   | }[num_metadata];               |
     *   |--------------------------------|
     */
    uint64_t header_size = 2*sizeof(uint64_t); /*format_id, num_metadata*/
    size_t idx = 0;
    for (auto& i : plugins) {
        this->view_segment(&tmp_in, ("stage-" + std::to_string(idx++)).c_str());
        ec = i->compress(&tmp_in, &tmp_out);
        if(ec) {
            set_error(ec, i->error_msg());
            if(ec > 0) return ec;
        }
        metadata.emplace_back(pressio_data::empty(tmp_in.dtype(), tmp_in.dimensions()));
        header_size += (
                sizeof(uint64_t)+                         /*num_dim*/
                sizeof(uint64_t)+                         /*type*/
                sizeof(uint64_t)* tmp_in.num_dimensions() /*dim*/
        ); /*stage_dims*/
        std::swap(tmp_in, tmp_out);
    }
    //tmp_in now holds the output; move to host to add header
    tmp_in = domain_manager().make_readable(domain_plugins().build("malloc"), std::move(tmp_in));

    //output needs to be on the host to add the header
    *output = pressio_data::owning(
            pressio_byte_dtype,
            {tmp_in.size_in_bytes() + header_size}
            );
    uint64_t* metadata_ptr = static_cast<uint64_t*>(output->data());
    metadata_ptr[0] = 1; /*version*/

    //bit_cast to int64_t
    const uint64_t num_metadata =metadata.size();
    memcpy(&metadata_ptr[1], &num_metadata, sizeof(int64_t));

    size_t output_idx = 2;
    for (auto const& m : metadata) {
        metadata_ptr[output_idx++] = m.num_dimensions(); /*num_dims*/
        metadata_ptr[output_idx++] = static_cast<uint64_t>(m.dtype()); /*type*/
        for (size_t i = 0; i < m.num_dimensions(); ++i) {
            metadata_ptr[output_idx++] = m.get_dimension(i); /*dim*/
        }
    }
    memcpy(static_cast<uint8_t*>(output->data()) + header_size, tmp_in.data(), tmp_in.size_in_bytes());

    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* output) override
  {
      //decode metadata
      //move to host to read header
      pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
      const uint64_t* metadata_ptr = static_cast<uint64_t*>(input.data());
      uint64_t idx = 0;
      const uint64_t format_id = metadata_ptr[idx++];
      std::vector<pressio_data> outputs;
      switch(format_id) {
          case 1:
              {
                  const int64_t num_metadata = (reinterpret_cast<const int64_t*>(metadata_ptr))[idx++];
                  for (int64_t i = 0; i < num_metadata; ++i) {
                      const uint64_t num_dims = metadata_ptr[idx++];
                      const pressio_dtype dtype = static_cast<pressio_dtype>(metadata_ptr[idx++]);
                      std::vector<size_t> dims(num_dims);
                      for (size_t d = 0; d < num_dims; ++d) {
                          dims[d] = metadata_ptr[idx++];
                      }
                      outputs.emplace_back(
                              pressio_data::empty(
                                    dtype, dims
                                  )
                              );
                  }
                  const size_t header_size_in_bytes = sizeof(uint64_t)*idx;
                  const size_t data_size_in_bytes = input.size_in_bytes() - header_size_in_bytes;
                  pressio_data tmp_in = pressio_data::nonowning(
                      pressio_byte_dtype,
                      static_cast<uint8_t *>(input.data()) + header_size_in_bytes,
                      {data_size_in_bytes});

                  //now decode them in reverse
                  if(num_metadata == 0) { // don't go negative
                      *output = std::move(tmp_in);
                      return 0;
                  }
                  for (int64_t i = num_metadata - 1; i >= 0; i--) {
                      pressio_data tmp_out = pressio_data::owning(outputs[i].dtype(), outputs[i].dimensions());
                      int ec = plugins[i]->decompress(&tmp_in, &tmp_out);
                      if(ec) {
                          set_error(ec, plugins[i]->error_msg());
                          if(ec > 0) {
                              return ec;
                          }
                      }
                      std::swap(tmp_in, tmp_out);
                  }
                  //now tmp_in has the final data, so move it into place
                  *output = std::move(tmp_in);
              }
              break;
          default:
              return set_error(1, "unrecognized header version");
      }
      return 0;
  }

  void set_name_impl(std::string const& name) override {
      set_names_many(name, plugins, names);
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "pipeline"; }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts;
    for (auto const& i : plugins) {
        opts.copy_from(i->get_metrics_results());
    }
    return opts;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<pipeline_compressor_plugin>(*this);
  }

  std::vector<std::string> ids, names;
  std::vector<pressio_compressor> plugins;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "pipeline", []() {
  return compat::make_unique<pipeline_compressor_plugin>();
});

} }

