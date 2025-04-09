
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/domain_manager.h"
#include "cleanup.h"
#include "timer.h"
#include "api/MSz.h"
#include <string>

namespace libpressio { namespace msz_ns {
    using namespace std::string_literals;

class msz_compressor_plugin : public libpressio_compressor_plugin {
public:
    static constexpr size_t HEADER_LEN = 2;
    static constexpr size_t EDITS_SIZE = 1;
    static constexpr size_t COMPRESSED_SIZE = 0;
    static constexpr int FULL_CONNECTION_TYPE = 1;
    static constexpr int PIECEWISE_CONNECTION_TYPE = 0;
  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "msz:compressor", compressor_id, compressor);
    set(options, "msz:preservation_options",preservation_options);
    set(options, "msz:connectivity_type",connectivity_type);
    set(options, "msz:accelerator", accelerator);
    set_type(options, "msz:preservation_options_str", pressio_option_charptr_array_type);
    set_type(options, "msz:connectivity_type_str",pressio_option_charptr_type);
    set_type(options, "msz:accelerator_str", pressio_option_charptr_type);

    set(options, "pressio:rel", rel_error_bound);
    set(options, "msz:rel_error_bound", rel_error_bound);

    set(options, "msz:cuda_device_id",device_id);
    set(options, "pressio:nthreads",num_threads);
    set(options, "msz:omp_num_threads",num_threads);

    set(options, "msz:count_faults",count_faults);
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "experimental");
    std::vector<std::string> invalidations {}; 
    std::vector<pressio_configurable const*> invalidation_children {&*compressor}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", invalidation_children, invalidations));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", invalidation_children, invalidations));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", invalidation_children, invalidations));
    set(options, "pressio:highlevel", get_accumulate_configuration("pressio:highlevel", invalidation_children, std::vector<std::string>{}));

    set(options, "msz:preservation_options_str",std::vector<std::string>{"min"s, "max"s, "path"s});
    set(options, "msz:connectivity_type_str",std::vector<std::string>{"piecewise"s, "full"s});
    set(options, "msz:accelerator_str", std::vector<std::string>{"cpu"s, "openmp"s, "cuda"s});
    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "msz:compressor", compressor_id, compressor);
    set(options, "pressio:description", R"()");
    set(options, "msz:preservation_options","what feature to preserve");
    set(options, "msz:connectivity_type", "how does the points connect to adjacent points");
    set(options, "msz:accelerator", "what hardware to use");
    set(options, "msz:preservation_options_str", "what feature to preserve");
    set(options, "msz:connectivity_type_str", "how does the points connect to adjacent points");
    set(options, "msz:accelerator_str", "what hardware to use");
    set(options, "msz:rel_error_bound", "value range relative error bound");
    set(options, "msz:cuda_device_id","what cuda device to use");
    set(options, "pressio:nthreads","how many threads to use");
    set(options, "msz:omp_num_threads","how many threads to use");
    set(options, "msz:count_faults","count faults on compression; adds overhead");
    set(options, "msz:num_false_labels", "number of false segmentation labels");
    set(options, "msz:num_false_min", "number of false minima after decompression");
    set(options, "msz:num_false_max", "number of false maxima after decompression");
    set(options, "msz:time_count_faults", "time to count faults in ms");
    set(options, "msz:time_derive_edits", "time to derive edits in ms");
    set(options, "msz:time_compress_edits", "time to compress edits in ms");
    set(options, "msz:time_compressed_to_outputs", "time to copy data to outputs in ms");
    set(options, "msz:time_apply_edits", "time to apply edits to decompressed data in ms");
    set(options, "msz:time_decompress_edits", "time to decompress edits in ms");
    set(options, "msz:num_edits", "number of edits");
    set(options, "msz:edits_size", "the size of the compressed edits");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "msz:compressor", compressor_plugins(), compressor_id, compressor);

    {
        std::vector<std::string> tmp;
        if(get(options, "msz:preservation_options_str", &tmp) == pressio_options_key_set) {
            uint32_t new_setting = 0;
            if(std::find(tmp.begin(), tmp.end(), "min"s) != tmp.end()) new_setting |= MSZ_PRESERVE_MIN;
            if(std::find(tmp.begin(), tmp.end(), "max"s) != tmp.end()) new_setting |= MSZ_PRESERVE_MAX;
            if(std::find(tmp.begin(), tmp.end(), "path"s) != tmp.end()) new_setting |= MSZ_PRESERVE_PATH;
            preservation_options = new_setting;
        }
    }
    {
        std::string tmp;
        if(get(options, "msz:connectivity_type_str",&tmp) == pressio_options_key_set) {
            uint32_t new_setting = 0;
            if(tmp == "piecewise") new_setting = 0;
            if(tmp == "full") new_setting = 1;
            connectivity_type = new_setting;
        }
        if(get(options, "msz:accelerator_str", &tmp) == pressio_options_key_set) {
            uint32_t new_setting = 0;
            if(tmp == "cpu") new_setting = MSZ_ACCELERATOR_NONE;
            if(tmp == "omp") new_setting = MSZ_ACCELERATOR_OMP;
            if(tmp == "cuda") new_setting = MSZ_ACCELERATOR_CUDA;
            accelerator = new_setting;
        }
    }

    get(options, "msz:preservation_options",&preservation_options);
    get(options, "msz:connectivity_type",&connectivity_type);
    get(options, "msz:accelerator", &accelerator);
    get(options, "msz:count_faults", &count_faults);

    if(preservation_options & MSZ_PRESERVE_MIN & MSZ_PRESERVE_PATH & ~MSZ_PRESERVE_MAX ) {
        return set_error(1, "both preserve min and path cannot be set together without max");
    }
    else if(preservation_options & MSZ_PRESERVE_MAX & MSZ_PRESERVE_PATH & ~MSZ_PRESERVE_MIN) {
        return set_error(1, "both preserve max and path cannot be set together without min");
    }

    if(connectivity_type == FULL_CONNECTION_TYPE && preservation_options & MSZ_PRESERVE_PATH) {
        return set_error(1, "cannot use full connection type with preserve path");
    }

    if(get(options, "msz:rel_error_bound", &rel_error_bound) == pressio_options_key_set) {}
    else get(options, "pressio:rel", &rel_error_bound);

    get(options, "msz:cuda_device_id", &device_id);

    if(get(options, "msz:omp_num_threads",&num_threads) == pressio_options_key_set) {
        //intentional no-op
    }
    else if(get(options, "pressio:nthreads",&num_threads) == pressio_options_key_set) {
        accelerator = MSZ_ACCELERATOR_OMP;
    }
    return 0;
  }

  int compress_impl(const pressio_data* real_input,
                    struct pressio_data* real_output) override
  {
    pressio_data output = pressio_data::empty(pressio_byte_dtype, {});
    if(compressor->compress(real_input, &output) > 0) {
        return set_error(compressor->error_code(), compressor->error_msg());
    }

    //MSz on compression needs the decompressed data to make its patches
    pressio_data decompressed = pressio_data::owning(*real_input, domain_plugins().build("malloc"));
    if(compressor->decompress(&output, &decompressed) > 0) {
        return set_error(compressor->error_code(), compressor->error_msg());
    }

    auto input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    input = input.cast(pressio_double_dtype);
    decompressed = domain_manager().make_readable(domain_plugins().build("malloc"), std::move(decompressed));
    decompressed = decompressed.cast(pressio_double_dtype);
    auto msz_dims = input.normalized_dims(3, 1);

    if(count_faults) {
        int32_t tmp_num_false_min, tmp_num_false_max, tmp_num_false_labels;

        start(timers.count_faults);
        int status = MSz_count_faults(
                static_cast<double*>(input.data()),
                static_cast<double*>(decompressed.data()),
                tmp_num_false_min, tmp_num_false_max, tmp_num_false_labels,
                connectivity_type,
                msz_dims[0], msz_dims[1], msz_dims[2],
                accelerator,
                device_id,
                num_threads
                );
        if(status != MSZ_ERR_NO_ERROR) {
            return msz_error_msg(status);
        }
        num_false_min = tmp_num_false_min;
        num_false_max = tmp_num_false_max;
        num_false_labels = tmp_num_false_labels;
        stop(timers.count_faults);
    } else {
        num_false_min = std::nullopt;
        num_false_max = std::nullopt;
        num_false_labels = std::nullopt;
    }

    int num_edits = 0;
    MSz_edit_t* edits = nullptr;
    auto cleanup_edits = make_cleanup([&]{free(edits);});
    start(timers.derive_edits);
    int status = MSz_derive_edits(
        static_cast<double*>(input.data()),
        static_cast<double*>(decompressed.data()),
        nullptr,
        num_edits, &edits,
        preservation_options,
        connectivity_type,
        msz_dims[0], msz_dims[1], msz_dims[2],
        rel_error_bound,
        accelerator,
        device_id,
        num_threads
    );
    stop(timers.derive_edits);
    this->num_edits = num_edits;
    if(status != MSZ_ERR_NO_ERROR) {
        return msz_error_msg(status);
    }

    char* compressed_buffer = nullptr;
    auto cleanup_compressed_buffer =  make_cleanup([&]{if(compressed_buffer != nullptr)free(compressed_buffer);});
    size_t compressed_buffer_size = 0;
    if(num_edits != 0){
        start(timers.compress_edits);
        int status = MSz_compress_edits_zstd(
                num_edits,
                edits,
                &compressed_buffer,
                compressed_buffer_size
                );
        stop(timers.compress_edits);
        this->edits_size = compressed_buffer_size;
        if (status != MSZ_ERR_NO_ERROR) {
            return msz_error_msg(status);
        }
    }

    start(timers.compressed_to_output);
    *real_output = pressio_data::owning(pressio_byte_dtype, {2*sizeof(uint64_t) + output.size_in_bytes() + compressed_buffer_size});
    uint8_t* ptr = static_cast<uint8_t*>(real_output->data());
    uint64_t o_size = static_cast<uint64_t>(output.size_in_bytes());
    memcpy(ptr, &o_size, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, &compressed_buffer_size, sizeof(uint64_t)); ptr += sizeof(uint64_t);
    memcpy(ptr, static_cast<uint8_t*>(output.data()), output.size_in_bytes()); ptr += output.size_in_bytes();
    if(num_edits != 0) {
        memcpy(ptr, compressed_buffer, compressed_buffer_size);
    }
    stop(timers.compressed_to_output);


    return 0;
  }

  int decompress_impl(const pressio_data* real_input,
                      struct pressio_data* output) override
  {
    auto input = domain_manager().make_readable(domain_plugins().build("malloc"), *real_input);
    auto header = static_cast<uint64_t*>(input.data());
    pressio_data compressed = pressio_data::nonowning(pressio_byte_dtype, header+HEADER_LEN, {header[COMPRESSED_SIZE]});
    char* compressed_edits = reinterpret_cast<char*>(((uint8_t*)header+HEADER_LEN)+header[COMPRESSED_SIZE]);

    compressor->decompress(&compressed, output);
    *output = domain_manager().make_readable(domain_plugins().build("malloc"), std::move(*output));
    pressio_dtype output_dtype = output->dtype();
    *output = output->cast(pressio_double_dtype);

    if(header[EDITS_SIZE] != 0) {
        int num_edits  = 0;
        MSz_edit_t* edits = nullptr;
        auto cleanup_edits = make_cleanup([&]{free(edits);});
        start(timers.decompress_edits);
        MSz_decompress_edits_zstd(
                compressed_edits,
                header[1],
                num_edits,
                &edits
                );
        stop(timers.decompress_edits);


        auto msz_dims = output->normalized_dims(3, 1);
        start(timers.apply_edits);
        MSz_apply_edits(
                static_cast<double*>(output->data()),
                num_edits, edits,
                msz_dims[0], msz_dims[1], msz_dims[2],
                accelerator
                );
        stop(timers.apply_edits);
        *output = output->cast(output_dtype);
    }

    return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 2; }
  const char* version() const override { return "0.0.2"; }
  const char* prefix() const override { return "msz"; }

  pressio_options get_metrics_results_impl() const override {
    auto metrics = compressor->get_metrics_results();
    set(metrics, "msz:num_false_labels", num_false_labels);
    set(metrics, "msz:num_false_min", num_false_min);
    set(metrics, "msz:num_false_max", num_false_max);
    set(metrics, "msz:time_count_faults", elapsed(timers.count_faults));
    set(metrics, "msz:time_derive_edits", elapsed(timers.derive_edits));
    set(metrics, "msz:time_compress_edits", elapsed(timers.compress_edits));
    set(metrics, "msz:time_compressed_to_outputs", elapsed(timers.compressed_to_output));
    set(metrics, "msz:time_apply_edits",   elapsed(timers.apply_edits));
    set(metrics, "msz:time_decompress_edits", elapsed(timers.decompress_edits));
    set(metrics, "msz:num_edits", num_edits);
    set(metrics, "msz:edits_size", edits_size);
    return metrics;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<msz_compressor_plugin>(*this);
  }

  void set_name_impl(std::string const& new_name) override {
      if(!new_name.empty()) compressor->set_name(new_name + "/casted");
      else compressor->set_name("");
  }

  uint32_t preservation_options = 0;
  uint32_t connectivity_type = PIECEWISE_CONNECTION_TYPE;
  double rel_error_bound = 1e-3;
  int32_t accelerator = MSZ_ACCELERATOR_NONE;
  int32_t device_id = 0;
  uint32_t num_threads = 0;
  bool count_faults = false;


  std::string compressor_id = "noop";
  pressio_compressor compressor = compressor_plugins().build("noop");

  //metrics
  std::optional<int32_t> num_false_min, num_false_max, num_false_labels, edits_size, num_edits;
  struct {
      utils::timer count_faults;
      utils::timer derive_edits;
      utils::timer compress_edits;
      utils::timer compressed_to_output;

      utils::timer apply_edits;
      utils::timer decompress_edits;
  } timers;

  int msz_error_msg(int status) {
        std::string msg;
        switch(status) {
            case MSZ_ERR_INVALID_INPUT: msg = "invalid input"; break;
            case MSZ_ERR_INVALID_CONNECTIVITY_TYPE: msg = "invalid connectivity"; break;
            case MSZ_ERR_NO_AVAILABLE_GPU: msg = "no GPU"; break;
            case MSZ_ERR_OUT_OF_MEMORY: msg = "out of memory"; break;
            case MSZ_ERR_UNKNOWN_ERROR: msg = "unknown error"; break;
            case MSZ_ERR_EDITS_COMPRESSION_FAILED: msg = "edits compression failed"; break;
            case MSZ_ERR_EDITS_DECOMPRESSION_FAILED:msg = "edits decompression failed"; break;
            case MSZ_ERR_NOT_IMPLEMENTED: msg = "not implemented"; break;
            case MSZ_ERR_INVALID_THREAD_COUNT:msg = "invalid thread count"; break;
        }
        return set_error(2, msg);
  }
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "msz", []() {
  return compat::make_unique<msz_compressor_plugin>();
});

} }

