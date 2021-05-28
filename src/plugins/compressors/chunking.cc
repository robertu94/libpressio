#include <sstream>
#include "libpressio_ext/cpp/data.h" //for access to pressio_data structures
#include "libpressio_ext/cpp/compressor.h" //for the libpressio_compressor_plugin class
#include "libpressio_ext/cpp/options.h" // for access to pressio_options
#include "libpressio_ext/cpp/pressio.h" //for the plugin registries
#include "pressio_options.h"
#include "pressio_data.h"
#include "pressio_compressor.h"
#include "std_compat/memory.h"
#include "std_compat/numeric.h"
#include "std_compat/functional.h"

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
      set(options, "chunking:size", pressio_data(chunk_size.begin(), chunk_size.end()));
      return options;
    }

    struct pressio_options get_configuration_impl() const override {
      struct pressio_options options;
      int compressor_thread_safety=pressio_thread_safety_single;
      compressor->get_configuration().get("pressio:thread_safe", &compressor_thread_safety);
      set(options, "pressio:thread_safe", static_cast<int>(compressor_thread_safety));
      set(options, "pressio:stability", "experimental");
      return options;
    }

    struct pressio_options get_documentation_impl() const override {
      struct pressio_options options;
      set(options, "pressio:description", R"(Chunks a larger dataset into smaller datasets for parallel compression)");
      set_meta_docs(options, "chunking:compressor", "compressor to use after chunking", compressor);
      set(options, "chunking:size", "size of the chunks to use");
      return options;
    }

    int set_options_impl(struct pressio_options const& options) override {
      get_meta(options, "chunking:compressor", compressor_plugins(), compressor_id, compressor);
      pressio_data d;
      if (get(options, "chunking:size", &d) == pressio_options_key_set) {
        chunk_size = d.to_vector<size_t>();
      }

      return 0;
    }


    void set_name_impl(std::string const& new_name) override {
      compressor->set_name(new_name + "/" + compressor->prefix());
    }


    int compress_impl(const pressio_data *input, struct pressio_data* output) override {
      if(not check_valid_dims(input)) return set_error(1, "chunks must currently be a multiple of input size");
      if(not check_contigous(input)) return set_error(2, "chunks must currently be contiguous within the input");

      //partition data into chunks
      size_t num_chunks = compute_num_chunks(input);
      size_t stride = std::accumulate(chunk_size.begin(), chunk_size.end(), pressio_dtype_size(input->dtype()), compat::multiplies<>{});
      std::vector<pressio_data> inputs; 
      std::vector<pressio_data> outputs; 
      std::vector<pressio_data*> inputs_ptr; 
      std::vector<pressio_data*> outputs_ptr; 
      inputs.reserve(num_chunks);
      outputs.reserve(num_chunks);
      inputs_ptr.reserve(num_chunks);
      outputs_ptr.reserve(num_chunks);
      std::vector<size_t> empty_dims = {};
      auto* ptr = reinterpret_cast<unsigned char*>(input->data());
      for (size_t i = 0; i < num_chunks; ++i) {
        inputs.emplace_back(pressio_data::nonowning(input->dtype(), ptr+(i*stride), chunk_size));
        inputs_ptr.emplace_back(&inputs.back());
        outputs.emplace_back(pressio_data::empty(pressio_byte_dtype, empty_dims));
        outputs_ptr.emplace_back(&outputs.back());
      }

      //run the child compressor on the chunks
      int rc = compressor->compress_many(
          inputs_ptr.data(),
          inputs_ptr.size(),
          outputs_ptr.data(),
          outputs_ptr.size()
          );

      //compute metadata sizes
      size_t total_compsize = std::accumulate(
          std::begin(outputs),
          std::end(outputs),
          static_cast<size_t>(0),
          [](size_t acc, pressio_data const& data) {
            return acc+ data.size_in_bytes();
          });
      size_t header_size = sizeof(uint64_t)*(outputs.size() + 1);

      *output = pressio_data::owning(pressio_byte_dtype, {header_size + total_compsize});

      //write header
      unsigned char* outptr = reinterpret_cast<unsigned char*>(output->data());
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
      return rc;
    }

    int decompress_impl(const pressio_data *input, struct pressio_data* output) override {
      //read in the header
      unsigned char* inptr = reinterpret_cast<unsigned char*>(input->data());
      uint64_t* inptr64 = reinterpret_cast<uint64_t*>(input->data());
      size_t n_buffers = *inptr64;
      const size_t header_size = sizeof(uint64_t) *(n_buffers+1);
      std::vector<uint64_t> sizes(inptr64+1, inptr64+(1+n_buffers));

      //create the buffers to decompress
      std::vector<pressio_data> inputs;
      std::vector<pressio_data> outputs;
      std::vector<pressio_data*> inputs_ptr;
      std::vector<pressio_data*> outputs_ptr;
      inputs.reserve(n_buffers);
      inputs_ptr.reserve(n_buffers);
      outputs.reserve(n_buffers);
      outputs_ptr.reserve(n_buffers);
      size_t accum_size = header_size;
      for (size_t i = 0; i < n_buffers; ++i) {
        inputs.emplace_back(pressio_data::nonowning(pressio_byte_dtype, inptr+accum_size, {sizes[i]}));
        outputs.emplace_back(pressio_data::owning(output->dtype(), chunk_size));
        inputs_ptr.emplace_back(&inputs.back());
        outputs_ptr.emplace_back(&outputs.back());
        accum_size+=sizes[i];
      }

      //run the decompressor
      int rc = compressor->decompress_many(
          inputs_ptr.data(),
          inputs_ptr.size(),
          outputs_ptr.data(),
          outputs_ptr.size()
          );

      //join the buffers
      unsigned char* outptr = reinterpret_cast<unsigned char*>(output->data());
      size_t accum_size_out = 0;
      const size_t stride_in_bytes = std::accumulate( std::begin(chunk_size), std::end(chunk_size), static_cast<size_t>(pressio_dtype_size(output->dtype())), compat::multiplies<>{});
      for (size_t i = 0; i < n_buffers; ++i) {
        memcpy(outptr+accum_size_out, outputs[i].data(), stride_in_bytes);
        accum_size_out += stride_in_bytes;
      }

      return rc;
    }


    //the author of SZauto does not release their version info.
    int major_version() const override {
      return 0; 
    }
    int minor_version() const override {
      return 0;
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
          [](size_t dim, size_t chunk){ return dim % chunk == 0; },
          [](bool current, bool next){ return current && next; }
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
      auto const& dims = input->dimensions();
      return compat::transform_reduce(
          dims.begin(),
          dims.end(),
          chunk_size.begin(),
          1,
          compat::multiplies<>{},
          [](size_t dim, size_t chunk) { return dim/chunk; }
          );
    }


    std::vector<size_t> chunk_size;
    std::string chunking_version;
    pressio_compressor compressor = compressor_plugins().build("noop");
    std::string compressor_id = "noop";
};

static pressio_register compressor_chunking_plugin(compressor_plugins(), "chunking", [](){return compat::make_unique<chunking_plugin>(); });
