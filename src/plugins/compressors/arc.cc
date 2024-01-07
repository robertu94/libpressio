#include <chrono>
#include "std_compat/memory.h"
#include "libpressio_ext/cpp/compressor.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"

#include <arc.h>

namespace libpressio { namespace arc_ns {

  struct arc_init_state {
    arc_init_state() {
      arc_init(omp_get_max_threads());
    }
    ~arc_init_state() {
      arc_close();
    }
  };

  std::mutex arc_init_lock;
  std::shared_ptr<arc_init_state> get_arc_init_state()  {
    std::lock_guard<std::mutex> guard(arc_init_lock);
    static std::weak_ptr<arc_init_state> handle;
    std::shared_ptr<arc_init_state> sp_handle;
    if((sp_handle = handle.lock())) {
      return sp_handle;
    } else {
      sp_handle = std::make_shared<arc_init_state>();
      handle = sp_handle;
      return sp_handle;
    }
  }

class arc_compressor_plugin : public libpressio_compressor_plugin {
public:
  arc_compressor_plugin(std::shared_ptr<arc_init_state>&& init_state): init_state(init_state) {}

  struct pressio_options get_options_impl() const override
  {
    struct pressio_options options;
    set_meta(options, "arc:compressor", impl_id, impl);
    set(options, "arc:memory_constraint", memory_constraint);
    set(options, "arc:throughput_constraint", throughput_constraint);
    set(options, "arc:do_save", do_save);
    set(options, "arc:ecc_options", pressio_data(ecc_choices.begin(), ecc_choices.end()));
    return options;
  }

  struct pressio_options get_configuration_impl() const override
  {
    struct pressio_options options;
    set_meta_configuration(options, "arc:compressor", compressor_plugins(), impl);
    set(options, "pressio:thread_safe", get_threadsafe(*impl));
    set(options, "pressio:stability", "experimental");

    std::vector<std::string> configs {"arc:memory_constraint", "arc:throughput_constraint", "arc:ecc_options"}; 
    set(options, "predictors:error_dependent", get_accumulate_configuration("predictors:error_dependent", {&*impl}, {}));
    set(options, "predictors:error_agnostic", get_accumulate_configuration("predictors:error_agnostic", {&*impl}, configs));
    set(options, "predictors:runtime", get_accumulate_configuration("predictors:runtime", {&*impl}, configs));

    return options;
  }

  struct pressio_options get_documentation_impl() const override
  {
    struct pressio_options options;
    set_meta_docs(options, "arc:compressor", "compressor to protect using ARC", impl);
    set(options, "pressio:description", R"(ARC is an automatic resiliency library designed to
        provide security to lossy compressed data or other uint8_t data arrays. Through minor
        user input and a single short training period, ARC determines the best ECC method for
        the data, applies it, and returns the encoded results to the user.)");
    set(options, "arc:memory_constraint", "set the memory constraint");
    set(options, "arc:throughput_constraint", "set the throughput constraint");
    set(options, "arc:do_save", "save configuration after configuration");
    set(options, "arc:ecc_options", "ecc_options available to ARC");
    return options;
  }


  int set_options_impl(struct pressio_options const& options) override
  {
    get_meta(options, "arc:compressor", compressor_plugins(), impl_id, impl);
    get(options, "arc:memory_constraint", &memory_constraint);
    get(options, "arc:throughput_constraint", &throughput_constraint);
    get(options, "arc:do_save", &do_save);
    pressio_data tmp;
    if (get(options, "arc:ecc_choices", &tmp) == pressio_options_key_set) {
      ecc_choices = tmp.to_vector<int>();
    }
    return 0;
  }

  int compress_impl(const pressio_data* input,
                    struct pressio_data* output) override
  {
    int rc = 0;
    if((rc = impl->compress(input, output)) > 0) {
      return set_error(impl->error_code(), impl->error_msg());
    } else if(rc < 0) {
      set_error(impl->error_code(), impl->error_msg());
    }

    uint8_t* arc_encoded;
    uint32_t arc_encoded_size;
    auto begin = std::chrono::steady_clock::now();
    auto err = arc_encode(
        static_cast<unsigned char*>(output->data()),
        output->size_in_bytes(),
        memory_constraint,
        throughput_constraint,
        ecc_choices.data(), static_cast<int>(ecc_choices.size()),
        &arc_encoded, &arc_encoded_size);
    auto end = std::chrono::steady_clock::now();
    encode_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    (void)err; //TODO what to do here?

    *output = pressio_data::move(
        pressio_byte_dtype,
        arc_encoded,
        {arc_encoded_size},
        pressio_data_libc_free_fn,
        nullptr
        );

    return 0;
  }

  int decompress_impl(const pressio_data* input,
                      struct pressio_data* output) override
  {
    uint8_t* arc_decoded;
    uint32_t arc_decoded_size;
    auto begin = std::chrono::steady_clock::now();
    auto err = arc_decode(static_cast<uint8_t*>(input->data()), input->size_in_bytes(), &arc_decoded, &arc_decoded_size);
    auto end = std::chrono::steady_clock::now();
    decode_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    (void)err; //TODO what to do here?

    pressio_data real_input = pressio_data::move(
        pressio_byte_dtype,
        arc_decoded,
        {arc_decoded_size},
        pressio_data_libc_free_fn,
        nullptr
        );
    int rc = impl->decompress(&real_input, output);

    if(rc > 0) {
      return set_error(impl->error_code(), impl->error_msg());
    } else if (rc < 0) {
      set_error(impl->error_code(), impl->error_msg());
    }

    if(do_save) {
      arc_save();
    }

    return 0;
  }

  int major_version() const override { return 0; }
  int minor_version() const override { return 0; }
  int patch_version() const override { return 1; }
  const char* version() const override { return "0.0.1"; }
  const char* prefix() const override { return "arc"; }

  void set_name_impl(std::string const& new_name) override {
    if(new_name != "") {
    impl->set_name(new_name + '/' + impl->prefix());
    } else {
    impl->set_name(new_name);
    }
  }
  std::vector<std::string> children_impl() const final {
      return { impl->get_name() };
  }

  pressio_options get_metrics_results_impl() const override {
    pressio_options opts = impl->get_metrics_results();
    set(opts, "arc:encode_time_ms", encode_time_ms);
    set(opts, "arc:decode_time_ms", decode_time_ms);
    return opts;
  }

  std::shared_ptr<libpressio_compressor_plugin> clone() override
  {
    return compat::make_unique<arc_compressor_plugin>(*this);
  }

  pressio_compressor impl = compressor_plugins().build("noop");
  std::string impl_id;

  int do_save = 0;
  double memory_constraint = ARC_ANY_SIZE;
  double throughput_constraint = ARC_ANY_BW;
  std::vector<int> ecc_choices {ARC_ANY_ECC};
  std::shared_ptr<arc_init_state> init_state;
  uint64_t encode_time_ms = 0, decode_time_ms = 0;
};

static pressio_register compressor_many_fields_plugin(compressor_plugins(), "arc", []() {
  return compat::make_unique<arc_compressor_plugin>(get_arc_init_state());
});

} }
