#include <sstream>
#include <chrono>
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "pressio_options.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "pressio_data.h"
#include "chunking_impl.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"
#include "std_compat/numeric.h"
#include "std_compat/functional.h"

namespace libpressio { namespace compressors { namespace chunking {

class chunking_plugin: public libpressio_compressor_plugin {
  public:
    chunking_plugin() {
      std::stringstream ss;
      ss << chunking_plugin::major_version() << "." << chunking_plugin::minor_version() << "." << chunking_plugin::patch_version();
      chunking_version = ss.str();
    };
    struct pressio_options get_options_impl() const override {
      struct pressio_options options;
      set_meta(options, "chunking:compressor", compressor_id, compressor);
      set(options, "pressio:nthreads", static_cast<uint32_t>(nthreads));
      set(options, "chunking:size", pressio_data(chunk_size.begin(), chunk_size.end()));
      set(options, "chunking:chunk_nthreads", nthreads);
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      set_meta_configuration(options, "chunking:compressor", compressor_plugins(), compressor);
      set(options, "pressio:thread_safe", get_threadsafe(*compressor));
      set(options, "pressio:stability", "experimental");
      
        std::vector<std::string> invalidations {"chunking:chunk_nthreads", "pressio:nthreads", "chunking:size"}; 
        std::vector<pressio_configurable const*> invalidation_children {&*compressor}; 
        
        set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, {}));
        set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, std::vector<std::string>{"chunking:size"}));
        set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
        set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{"pressio:nthreads"}));

    return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set_meta_docs(options, "chunking:compressor", "compressor to use after chunking", compressor);
      set(options, "pressio:description", R"(Chunks a larger dataset into smaller datasets for parallel compression)");
      set(options, "chunking:size", "size of the chunks to use");
      set(options, "chunking:chunk_nthreads", "number of threads to use to chunk the data");
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get_meta(options, "chunking:compressor", compressor_plugins(), compressor_id, compressor);

      uint32_t tmp;
      if(get(options, "pressio:nthreads", &tmp) == pressio_options_key_set) {
        if(tmp > 0) {
          nthreads = tmp;
        } else {
          return set_error(1,"nthreads must be positive");
        }
      }

      pressio_data d;
      if (get(options, "chunking:size", &d) == pressio_options_key_set) {
        chunk_size = d.to_vector<size_t>();
      }
      get(options, "chunking:chunk_nthreads", &nthreads);


      return 0;
    }

    pressio_options get_metrics_results_impl() const override {
      pressio_options mr = compressor->get_metrics_results();
      set(mr, "chunking:chunk_time", chunk_time);
      set(mr, "chunking:dechunk_time", dechunk_time);
      set(mr, "chunking:write_chunk_time", write_chunk_time);
      set(mr, "chunking:read_dechunk_time", read_dechunk_time);
      return mr;
    }


    void set_name_impl(std::string const& new_name) override {
      if(new_name != "") {
      compressor->set_name(new_name + "/" + compressor->prefix());
      } else {
      compressor->set_name(new_name);
      }
    }
  std::vector<std::string> children_impl() const final {
      return { compressor->get_name() };
  }


    int compress_impl(const pressio_data *real_input, struct pressio_data* real_output) override {
      auto chunk_begin = std::chrono::steady_clock::now();
      //partition data into chunks
      pressio_data tmp;
      size_t num_chunks = compute_num_chunks(real_input);
      std::vector<pressio_data> inputs; 
      std::vector<pressio_data> outputs; 
      std::vector<pressio_data const*> inputs_ptr; 
      std::vector<pressio_data*> outputs_ptr; 
      inputs.reserve(num_chunks);
      outputs.reserve(num_chunks);
      inputs_ptr.reserve(num_chunks);
      outputs_ptr.reserve(num_chunks);
      std::vector<size_t> empty_dims = {};
      size_t stride = std::accumulate(chunk_size.begin(), chunk_size.end(), pressio_dtype_size(real_input->dtype()), compat::multiplies<>{});
      if (num_chunks == 1) {
          //no copy to the host
          inputs_ptr.emplace_back(real_input);
          outputs.emplace_back(pressio_data::empty(pressio_byte_dtype, empty_dims));
          outputs_ptr.emplace_back(&outputs.back());
      } else if (check_contigous(real_input) and check_valid_dims(real_input)){
        //no copy to the host, everything is nonowning, so we don't need to move data here
        auto* ptr = reinterpret_cast<unsigned char*>(real_input->data());
        for (size_t i = 0; i < num_chunks; ++i) {
          inputs.emplace_back(pressio_data::nonowning(real_input->dtype(), ptr+(i*stride), chunk_size, real_input->domain()->domain_id()));
          inputs_ptr.emplace_back(&inputs.back());
          outputs.emplace_back(pressio_data::empty(pressio_byte_dtype, empty_dims));
          outputs_ptr.emplace_back(&outputs.back());
        }
      } else {
        //non-contigious, need to copy
        pressio_data input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
        tmp = libpressio::compressors::chunking::chunk_data(input, chunk_size, {{"nthreads", nthreads}});
        auto ptr = static_cast<uint8_t*>(tmp.data());
        for (size_t i = 0; i < num_chunks; ++i) {
          inputs.emplace_back(pressio_data::nonowning(real_input->dtype(), ptr+(i*stride), chunk_size, "malloc"));
          inputs_ptr.emplace_back(&inputs.back());
          outputs.emplace_back(pressio_data::empty(pressio_byte_dtype, empty_dims));
          outputs_ptr.emplace_back(&outputs.back());
        }
      }
      auto chunk_end = std::chrono::steady_clock::now();
      chunk_time = std::chrono::duration_cast<std::chrono::milliseconds>(chunk_end-chunk_begin).count();

      //run the child compressor on the chunks
      int rc = compressor->compress_many(
          inputs_ptr.data(),
          inputs_ptr.size(),
          outputs_ptr.data(),
          outputs_ptr.size()
          );

      if(rc > 0) {
        set_error(compressor->error_code(), compressor->error_msg());
        return rc;
      } else if(rc < 0) {
        set_error(compressor->error_code(), compressor->error_msg());
      }


      //compute metadata sizes
      auto write_chunk_begin = std::chrono::steady_clock::now();
      size_t total_compsize = std::accumulate(
          std::begin(outputs),
          std::end(outputs),
          static_cast<size_t>(0),
          [](size_t acc, pressio_data const& data) {
            return acc+ data.size_in_bytes();
          });
      size_t header_size = sizeof(uint64_t)*(outputs.size() + 1);

      pressio_data output =
          (real_output->has_data())
              ? (domain_manager().make_writeable(domain_plugins().build("malloc"),
                                                 std::move(*real_output)))
              : (pressio_data::owning(pressio_byte_dtype, {header_size + total_compsize}));

      //write header
      unsigned char* outptr = reinterpret_cast<unsigned char*>(output.data());
      uint64_t* outputs_ptr64 = reinterpret_cast<uint64_t*>(outptr);
      *outputs_ptr64 = outputs.size();
      outputs_ptr64++;
      std::transform(
          std::begin(outputs),
          std::end(outputs),
          outputs_ptr64,
          [](pressio_data const& data) {
            return static_cast<uint64_t>(data.size_in_bytes());
          });

      //write compressed data
      size_t accum_size = header_size;
      for (auto const& i : outputs) {
        memcpy(outptr+accum_size, i.data(), i.size_in_bytes());
        accum_size += i.size_in_bytes();
      }
      auto write_chunk_end = std::chrono::steady_clock::now();
      write_chunk_time = std::chrono::duration_cast<std::chrono::milliseconds>(write_chunk_end-write_chunk_begin).count();

      *real_output = std::move(output);
      return rc;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      //read in the header
      auto read_dechunk_begin = std::chrono::steady_clock::now();
      unsigned char* inptr = reinterpret_cast<unsigned char*>(input->data());
      uint64_t* inptr64 = reinterpret_cast<uint64_t*>(input->data());
      size_t n_buffers = *inptr64;
      const size_t header_size = sizeof(uint64_t) *(n_buffers+1);
      std::vector<uint64_t> sizes(inptr64+1, inptr64+(1+n_buffers));

      //create the buffers to decompress
      std::vector<pressio_data> inputs;
      std::vector<pressio_data> outputs;
      std::vector<pressio_data const*> inputs_ptr;
      std::vector<pressio_data*> outputs_ptr;
      inputs.reserve(n_buffers);
      inputs_ptr.reserve(n_buffers);
      outputs.reserve(n_buffers);
      outputs_ptr.reserve(n_buffers);
      size_t accum_size = header_size;
      for (size_t i = 0; i < n_buffers; ++i) {
        inputs.emplace_back(pressio_data::nonowning(pressio_byte_dtype, inptr+accum_size, {sizes[i]}));
        outputs.emplace_back(pressio_data::owning(output->dtype(), (chunk_size.empty() ? output->dimensions(): chunk_size )));
        inputs_ptr.emplace_back(&inputs.back());
        outputs_ptr.emplace_back(&outputs.back());
        accum_size+=sizes[i];
      }
      auto read_dechunk_end = std::chrono::steady_clock::now();
      read_dechunk_time = std::chrono::duration_cast<std::chrono::milliseconds>(read_dechunk_end-read_dechunk_begin).count();

      //run the decompressor
      int rc = compressor->decompress_many(
          inputs_ptr.data(),
          inputs_ptr.size(),
          outputs_ptr.data(),
          outputs_ptr.size()
          );
      if(rc) {
        set_error(rc, compressor->error_msg());
      }

      auto dechunk_begin = std::chrono::steady_clock::now();
      //join the buffers
      if(n_buffers == 1) {
        *output = std::move(outputs[0]);
      } else if(check_contigous(output) and check_valid_dims(output)) {
        unsigned char* outptr = reinterpret_cast<unsigned char*>(output->data());
        size_t accum_size_out = 0;
        const size_t stride_in_bytes = std::accumulate( std::begin(chunk_size), std::end(chunk_size), static_cast<size_t>(pressio_dtype_size(output->dtype())), compat::multiplies<>{});
        for (size_t i = 0; i < n_buffers; ++i) {
          memcpy(outptr+accum_size_out, outputs[i].data(), stride_in_bytes);
          accum_size_out += stride_in_bytes;
        }
      } else {
        size_t accum_size_out = 0;
        const size_t stride_in_bytes = std::accumulate( std::begin(chunk_size), std::end(chunk_size), static_cast<size_t>(pressio_dtype_size(output->dtype())), compat::multiplies<>{});
        pressio_data combined(pressio_data::owning(pressio_byte_dtype, {n_buffers * stride_in_bytes}));
        unsigned char* outptr = reinterpret_cast<unsigned char*>(combined.data());
        for (size_t i = 0; i < n_buffers; ++i) {
          memcpy(outptr+accum_size_out, outputs[i].data(), stride_in_bytes);
          accum_size_out += stride_in_bytes;
        }
        libpressio::compressors::chunking::restore_data(*output, combined, chunk_size, {{"nthreads", nthreads}});
      }
      auto dechunk_end = std::chrono::steady_clock::now();
      dechunk_time = std::chrono::duration_cast<std::chrono::milliseconds>(dechunk_end-dechunk_begin).count();

      return rc;
    }

    int major_version() const override {
      return 0; 
    }
    int minor_version() const override {
      return 1;
    }
    int patch_version() const override {
      return 0;
    }

    const char* version() const override {
      return chunking_version.c_str(); 
    }


    const char* prefix() const override {
      return "chunking";
    }

    std::shared_ptr<libpressio_compressor_plugin> clone() override {
      return compat::make_unique<chunking_plugin>(*this);
    }
  private:
    bool check_valid_dims(pressio_data const* input) const {
      auto const& dims = input->dimensions();
      if(dims.size() != chunk_size.size()) return false;
      //only support chunks without padding for now.
      return compat::transform_reduce(
          dims.begin(),
          dims.end(),
          chunk_size.begin(),
          true,
          [](bool current, bool next){ return current && next; },
          [](size_t dim, size_t chunk){ return dim % chunk == 0; }
          );
    }
    bool check_contigous(pressio_data const* input) const {
      auto const& dims = input->dimensions();
      bool mismatch = false;
      for (size_t i = 0; i < std::min(dims.size(), chunk_size.size()); ++i) {
        if(mismatch) {
          if (chunk_size[i] != 1) {
            return false;
          } 
        } else {
          mismatch = (dims[i] != chunk_size[i]);
        }
      }
      return true;
    }

    size_t compute_num_chunks(pressio_data const* input) const {
      if(chunk_size.empty()) {
        return 1;
      }
      auto const& dims = input->dimensions();
      return compat::transform_reduce(
          dims.begin(),
          dims.end(),
          chunk_size.begin(),
          1,
          compat::multiplies<>{},
          [](size_t dim, size_t chunk) {
            if(dim % chunk == 0) return dim/chunk; 
            else return dim/chunk+1;
          });
    }


    std::vector<size_t> chunk_size;
    std::string chunking_version;
    pressio_compressor compressor = compressor_plugins().build("noop");
    std::string compressor_id = "noop";
    compat::optional<uint64_t> chunk_time;
    compat::optional<uint64_t> read_dechunk_time;
    compat::optional<uint64_t> dechunk_time;
    compat::optional<uint64_t> write_chunk_time;
    uint64_t nthreads = 1;
};

pressio_register registration(compressor_plugins(), "chunking", [](){return compat::make_unique<chunking_plugin>(); });
} } }
