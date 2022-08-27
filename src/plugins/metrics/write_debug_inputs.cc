#include "pressio_data.h"
#include "pressio_compressor.h"
#include "pressio_options.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/printers.h"
#include "std_compat/memory.h"
#include <iostream>
#include <iomanip>
#include <mutex>

namespace libpressio { namespace write_debug_inputs_metrics_ns {

  std::mutex idx_mutex;
  uint64_t input_write_idx = 0;
  uint64_t compressed_write_idx = 0;
  uint64_t output_write_idx = 0;

class write_debug_inputs_plugin : public libpressio_metrics_plugin {
  enum class mode {
    input,
    compressed,
    output,
  };
  public:
    int end_compress_impl(struct pressio_data const* input, pressio_data const*, int) override {
      if(write_input) {
        std::string path = build_path(mode::input);
        if(display_paths) {
          std::cerr << std::quoted(path) << ' ' << *input << std::endl;
        }
        io->set_options({{"io:path", path}});
        io->write(input);
      }
      if(write_compressed) {
        std::string path = build_path(mode::compressed);
        if(display_paths) {
          std::cerr << std::quoted(path) << ' ' << *input << std::endl;
        }
        io->set_options({{"io:path", path}});
        io->write(input);
      }
      return 0;
    }

    int end_decompress_impl(struct pressio_data const* , pressio_data const* output, int) override {
      if(write_output) {
        std::string path = build_path(mode::output);
        if(display_paths) {
          std::cerr << std::quoted(path) << ' ' << *output << std::endl;
        }
        io->set_options({{"io:path", path}});
        io->write(output);
      }
      return 0;
    }
  
  struct pressio_options get_options() const override {
    pressio_options opts;
    set_meta(opts, "write_debug_inputs:io", io_format, io);
    set(opts, "write_debug_inputs:base_path", base_path);
    set(opts, "write_debug_inputs:display_paths", display_paths);
    set(opts, "write_debug_inputs:write_output", write_output);
    set(opts, "write_debug_inputs:write_input", write_input);
    set(opts, "write_debug_inputs:write_compressed", write_compressed);
    set_type(opts, "write_debug_inputs:write_all", pressio_option_bool_type);
    return opts;
  }
  int set_options(pressio_options const& opts) override {
    get_meta(opts, "write_debug_inputs:io", io_plugins(), io_format, io);
    get(opts, "write_debug_inputs:base_path", &base_path);
    get(opts, "write_debug_inputs:display_paths", &display_paths);
    {
      bool tmp;
      if(get(opts, "write_debug_inputs:write_all", &tmp) == pressio_options_key_set) {
        write_input = tmp;
        write_output = tmp;
        write_compressed = tmp;
      }
    }
    get(opts, "write_debug_inputs:write_output", &write_output);
    get(opts, "write_debug_inputs:write_input", &write_input);
    get(opts, "write_debug_inputs:write_compressed", &write_compressed);
    return 0;
  }


  struct pressio_options get_configuration() const override {
    pressio_options opts;
    set(opts, "pressio:stability", "stable");
    set(opts, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    return opts;
  }

  struct pressio_options get_documentation_impl() const override {
    pressio_options opt;
    set_meta_docs(opt, "write_debug_inputs:io", "format used for all data to be written", io);
    set(opt, "pressio:description", "write inputs to compress and decompress to disk for examination");
    set(opt, "write_debug_inputs:base_path", "base path where outputs will be written");
    set(opt, "write_debug_inputs:display_paths", "display paths where outputs will be written");
    set(opt, "write_debug_inputs:write_output", "write_output data");
    set(opt, "write_debug_inputs:write_input", "write_input data");
    set(opt, "write_debug_inputs:write_compressed", "write_compressed data");
    set(opt, "write_debug_inputs:write_all", "shortcut for write_input, write_output, and write_compressed");
    return opt;
  }

  pressio_options get_metrics_results(pressio_options const &) override {
    pressio_options opt;
    return opt;
  }

  std::unique_ptr<libpressio_metrics_plugin> clone() override {
    return compat::make_unique<write_debug_inputs_plugin>(*this);
  }
  const char* prefix() const override {
    return "write_debug_inputs";
  }

  private:

  std::string build_path(mode m) {
    std::lock_guard<std::mutex> guard(idx_mutex);
    switch(m) {
      case mode::input:
        return base_path + "input-" + std::to_string(input_write_idx++);
      case mode::compressed:
        return base_path + "compressed-" + std::to_string(compressed_write_idx++);
      case mode::output:
        return base_path + "output-" + std::to_string(output_write_idx++);
      default:
        throw std::logic_error("unexpected mode");
    }
  }

  static std::string safe_getenv(const char* key, std::string deflt) {
    const char* env = getenv(key);
    if (env != nullptr) {
      return std::string(env) + '/';
    } else {
      return deflt;
    }
  }

  pressio_io io = io_plugins().build("noop");
  std::string io_format = "noop";
  std::string base_path = safe_getenv("TMPDIR", "/tmp/");
  bool display_paths = false;
  bool write_input = false;
  bool write_compressed = false;
  bool write_output = false;
};

static pressio_register metrics_write_debug_inputs_plugin(metrics_plugins(), "write_debug_inputs", [](){ return compat::make_unique<write_debug_inputs_plugin>(); });
}}
