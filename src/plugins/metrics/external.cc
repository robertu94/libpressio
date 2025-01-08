#include "pressio_version.h"
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <unistd.h>
#include <chrono>
#include <iterator>
#include "pressio_data.h"
#include "pressio_options.h"
#include "pressio_compressor.h"
#include "libpressio_ext/io/pressio_io.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/io.h"
#include "std_compat/memory.h"
#include "std_compat/language.h"
#include "external_parse.h"

#if LIBPRESSIO_HAS_LIBUUID == 1
#include <uuid.h>
#else
#include <random>
#include <sstream>
#include <iomanip>
#endif

#include "libpressio_ext/launch/external_launch.h"

#if LIBPRESSIO_HAS_JSON
#include <nlohmann/json.hpp>
#include "libpressio_ext/cpp/json.h"
#endif

pressio_registry<std::unique_ptr<libpressio_launch_plugin>>& launch_plugins() {
  static pressio_registry<std::unique_ptr<libpressio_launch_plugin>> registry;
  return registry;
}

namespace libpressio { namespace external_metrics {


class external_metric_plugin : public libpressio_metrics_plugin {

  public:
    int begin_compress_impl(const struct pressio_data * input, struct pressio_data const * ) override {
      if((not use_many) and field_names.size() == 1) {
        input_data.resize(1);
        input_data.back() = pressio_data::clone(*input);
      }
      return 0;
    }

    int begin_compress_many_impl(compat::span<const pressio_data* const> const& inputs,
                                   compat::span<const pressio_data* const> const&) override {
      if(use_many or inputs.size() > 1) {
        input_data.resize(inputs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
          input_data[i] = pressio_data::clone(*inputs[i]);
        }
      }
      return 0;
    }

    int end_decompress_impl(struct pressio_data const*, struct pressio_data const* output, int ) override {
      if((not use_many) and field_names.size() == 1) {
        std::vector<const pressio_data*> input_ptrs(input_data.size());
        for (size_t i = 0; i < input_data.size(); ++i) {
          input_ptrs[i] = &input_data[i];
        }
        compat::span<const pressio_data* const> input_datasets{input_ptrs.data(), 1};
        compat::span<const pressio_data* const> output_datasets{&output, 1};
        run_external(input_datasets, output_datasets);
      }
      return 0;
    }

    int end_decompress_many_impl(compat::span<const pressio_data* const> const& ,
                                   compat::span<const pressio_data* const> const& outputs, int ) override {
      if(use_many or outputs.size() > 1) {
        std::vector<const pressio_data*> input_ptrs(input_data.size());
        for (size_t i = 0; i < input_data.size(); ++i) {
          input_ptrs[i] = &input_data[i];
        }
        compat::span<const pressio_data* const> input_datasets{input_ptrs.data(), input_ptrs.size()};
        run_external(input_datasets, outputs);
      }
      return 0;
    }

    struct pressio_options get_configuration_impl() const override {
      pressio_options opts;
      set_meta_many_configuration(opts, "external:io_format", io_plugins(), io_modules);
      set_meta_configuration(opts, "external:launch_method", launch_plugins(), launcher);
      set(opts, "pressio:stability", "unstable");
      set(opts, "pressio:thread_safe", pressio_thread_safety_multiple);
      set(opts, "predictors:requires_decompress", true);
      set(opts, "predictors:invalidate", std::vector<std::string>{"predictors:error_dependent"});
      return opts;
    }

    struct pressio_options get_documentation_impl() const override {
      pressio_options opt;
      set_meta_docs(opt, "external:launch_method", "launch methods to use with the external metric", launcher);
      set_meta_many_docs(opt, "external:io_format", "io formats used to write out data buffers", io_modules);
      set(opt, "pressio:description", "metrics module that launches an external task");
      set(opt, "external:stderr", "the stderr from the external process, used for error reporting");
      set(opt, "external:return_code", "the return code from the external process if it was launched");
      set(opt, "external:error_code", "error code, indicates problems with launching processes");
      set(opt, "external:runtime", "runtime of the external request, in seconds");
      set(opt, "external:command", "the command to use passed as a single string");
      set(opt, "external:suffix", "suffix to use in generated temporary file names");
      set(opt, "external:prefix", "prefix to use in generated temporary file names");
      set(opt, "external:fieldnames", "names of the fields used when generating arguments");
      set(opt, "external:workdir", "directory to launch the external process in");
      set(opt, "external:config_name", "string passed to the config_name option");
      set(opt, "external:use_many", "use the begin_many/end_many versions for launching the job when there is only one buffer");
      set(opt, "external:write_inputs", "disables the writing of all inputs");
      set(opt, "external:write_outputs", "disables the writing of all ouputs");

      return opt;
    }

    struct pressio_options get_options() const override {
      auto opt = pressio_options();
      set_meta(opt, "external:launch_method", launch_method, launcher);
      set_meta_many(opt, "external:io_format", io_formats, io_modules);
      set_type(opt, "external:command", pressio_option_charptr_type);
      set(opt, "external:suffix", suffixes);
      set(opt, "external:prefix", prefixes);
      set(opt, "external:fieldnames", field_names);
      set(opt, "external:workdir", workdir);
      set(opt, "external:config_name", config_name);
      set(opt, "external:use_many", use_many);
      set(opt, "external:write_inputs", write_inputs);
      set(opt, "external:write_outputs", write_outputs);
      return opt;
    }

    int set_options(pressio_options const& opt) override {
      pressio_options mopt = opt;
      std::string command;
      if(get(opt, "external:command", &command) == pressio_options_key_set) {
        std::istringstream iss(command);
        std::vector<std::string> commands((std::istream_iterator<std::string>(iss)), std::istream_iterator<std::string>());
        set(mopt, "external:commands", commands);
      }

      get_meta(mopt, "external:launch_method", launch_plugins(), launch_method, launcher);

      get(mopt, "external:suffix", &suffixes);
      get(mopt, "external:prefix", &prefixes);
      get(mopt, "external:fieldnames", &field_names);
      get(mopt, "external:workdir", &workdir);
      get(mopt, "external:config_name", &config_name);
      get(mopt, "external:use_many", &use_many);
      get(mopt, "external:write_inputs", &write_inputs);
      get(mopt, "external:write_outputs", &write_outputs);
      get_meta_many(opt, "external:io_format", io_plugins(), io_formats, io_modules);
      return 0;
    }

    pressio_options get_metrics_results(pressio_options const &)  override {
      if(results.size() == 0) {
        pressio_options ret;
        auto default_result = launcher->launch({});
        parse_result(default_result, ret, true, get_name(), duration);
        return ret;
      }
      return results;
    }

    void set_name_impl(std::string const& new_name) override {
      if (new_name != "") {
        for (size_t i = 0; i < std::min(io_modules.size(), field_names.size()); ++i) {
          io_modules[i]->set_name(new_name + "/" + field_names[i]);
        }
        launcher->set_name(new_name + "/" + launcher->prefix());
      } else {
        for (size_t i = 0; i < std::min(io_modules.size(), field_names.size()); ++i) {
          io_modules[i]->set_name(new_name);
        }
        launcher->set_name(new_name);
      }
    }
    std::vector<std::string> children() const final {
        std::vector<std::string> result;
        result.reserve(io_modules.size() + 1);
        for (size_t i = 0; i < io_modules.size(); ++i) {
            result.push_back(io_modules[i]->get_name());
        }
        result.push_back(launcher->get_name());
        return result;
    }

    std::unique_ptr<libpressio_metrics_plugin> clone() override {
      return compat::make_unique<external_metric_plugin>(*this);
    }

    const char* prefix() const override {
      return "external";
    }


  private:



    std::vector<std::string> build_command(std::vector<std::pair<std::string,std::string>> const& filenames, compat::span<const pressio_data* const> const& input_datasets) const {
      std::vector<std::string> full_command;
      full_command.emplace_back("--api");
      full_command.emplace_back("6");
      full_command.emplace_back("--config_name");
      full_command.emplace_back(config_name);

      constexpr int uuid_str_size = 36;
      char uuid_str[uuid_str_size + 1] = {0};
#if LIBPRESSIO_HAS_LIBUUID
      uuid_t uuid;
      uuid_generate(uuid);
      uuid_unparse(uuid, uuid_str);
#else
      constexpr int uuid_byte_size = 16;
      uint8_t uuid[uuid_byte_size] = {0};
      std::random_device hw_rng;
      std::seed_seq seed{hw_rng(), hw_rng()};
      std::mt19937 gen(seed);
      std::uniform_int_distribution<uint8_t> dist(0, std::numeric_limits<uint8_t>::max());
      std::generate_n(uuid, uuid_byte_size, [&]{return dist(gen);});
      std::stringstream ss;
      for (int i = 0; i < uuid_byte_size; ++i) {
          ss << std::hex << int(uuid[i]);
          if (i == 3 || i == 5 || i == 7 || i == 9) {
              ss << '-';
          }
      }
      auto s = ss.str();
      std::copy(s.begin(), s.end(), uuid_str);
#endif

      full_command.emplace_back("--eval_uuid");
      full_command.emplace_back(uuid_str);

      auto format_arg = [this](size_t i, std::string const& arg) {
        std::ostringstream ss;
        if(i >= field_names.size() || field_names[i].empty()) {
          ss << "--" << arg;
        } else {
          ss << "--" << field_names[i] << '_' << arg;
        }
        return ss.str();
      };
      for (size_t i = 0; i < filenames.size(); ++i) {
        auto const& input_path = filenames[i].first;
        auto const& decomp_path = filenames[i].second;
        auto const& input_data = input_datasets[i];
        full_command.emplace_back(format_arg(i, "input"));
        full_command.emplace_back(input_path);
        full_command.emplace_back(format_arg(i, "decompressed"));
        full_command.emplace_back(decomp_path);
        full_command.emplace_back(format_arg(i, "type"));
        switch(input_data->dtype()) {
          case pressio_float_dtype: full_command.emplace_back("float"); break;
          case pressio_double_dtype: full_command.emplace_back("double"); break;
          case pressio_bool_dtype: full_command.emplace_back("bool"); break;
          case pressio_int8_dtype: full_command.emplace_back("int8"); break;
          case pressio_int16_dtype: full_command.emplace_back("int16"); break;
          case pressio_int32_dtype:  full_command.emplace_back("int32"); break;
          case pressio_int64_dtype:  full_command.emplace_back("int64"); break;
          case pressio_uint8_dtype:  full_command.emplace_back("uint8"); break;
          case pressio_uint16_dtype: full_command.emplace_back("uint16"); break;
          case pressio_uint32_dtype: full_command.emplace_back("uint32"); break;
          case pressio_uint64_dtype: full_command.emplace_back("uint64"); break;
          case pressio_byte_dtype: full_command.emplace_back("byte"); break;
        }
        for (auto dim : input_data->dimensions()) {
          full_command.emplace_back(format_arg(i, "dim"));
          full_command.emplace_back(std::to_string(dim));
        }
        
      }
      return full_command;
    }

    void run_external(compat::span<const pressio_data* const> const& input_data, compat::span<const pressio_data* const> const& decompressed_data) {
      std::vector<int> fds;
      std::vector<std::pair<std::string,std::string>> filenames;
      auto get_or = [](std::vector<std::string> const& array, size_t index) -> compat::optional<std::string> {
        if(index < array.size()){
          return array[index];
        } else {
          return {};
        }
      };
      auto get = [](std::vector<pressio_io>& array, size_t index) -> pressio_io& {
        if(array.size() != 1){
          return array[index];
        } else {
          return array.front();
        }
      };

      for (size_t i = 0; i < std::min(input_data.size(), decompressed_data.size()); ++i) {
          //write uncompressed data to a temporary file
          std::string input_fd_name = get_or(prefixes, i).value_or("") + std::string(".pressioinXXXXXX") + get_or(suffixes, i).value_or("");
          int input_fd = mkstemps(&input_fd_name[0], get_or(suffixes, i).value_or("").size());
          char* resolved_input = realpath(input_fd_name.c_str(), nullptr);
          input_fd_name = resolved_input;
          free(resolved_input);
          if(write_inputs) {
              get(io_modules,i)->set_options({{"io:path", std::string(input_fd_name)}});
              get(io_modules,i)->write(input_data[i]);
          }

          //write decompressed data to a temporary file
          std::string output_fd_name = get_or(prefixes, i).value_or("") + std::string(".pressiooutXXXXXX") + get_or(suffixes, i).value_or("");
          int decompressed_fd = mkstemps(&output_fd_name[0], get_or(suffixes,i).value_or("").size());
          char* resolved_output = realpath(output_fd_name.c_str(), nullptr);
          output_fd_name = resolved_output;
          free(resolved_output);
          if(write_outputs) {
              get(io_modules,i)->set_options({{"io:path", std::string(output_fd_name)}});
              get(io_modules,i)->write(decompressed_data[i]);
          }

          filenames.emplace_back(input_fd_name, output_fd_name);
          fds.emplace_back(input_fd);
          fds.emplace_back(decompressed_fd);
      }

      //get the defaults
      auto default_result = launcher->launch({});
      parse_result(default_result, this->defaults, true, get_name(), duration);

      //build the command
      auto full_command = build_command(filenames, input_data);

      //run the external program
      auto start_time = std::chrono::high_resolution_clock::now();
      auto result = launcher->launch(full_command);
      auto end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration<double>(end_time - start_time).count();

      //parse the output
      size_t api_version =parse_result(result, this->results, false, get_name(), duration);

      //combine the results by setting defaults
      if(api_version >= 3) {
        for (auto const& default_v : defaults) {
          if (results.key_status(default_v.first) != pressio_options_key_set) {
            set(results, default_v.first, defaults.get(default_v.first));
          }
        }
      }

      //delete the temporary files
      std::for_each(std::begin(fds), std::end(fds), [](int fd){ close(fd);});
      std::for_each(std::begin(filenames), std::end(filenames),
          [](std::pair<std::string,std::string> const& file){
            unlink(file.first.c_str());
            unlink(file.second.c_str());
          });
    }

    int use_many = 0;
    std::vector<pressio_data> input_data;
    std::string workdir = ".";
    std::string launch_method = "forkexec";
    std::string config_name = "external";
    pressio_launcher launcher = launch_plugins().build("forkexec");
    std::vector<std::string> field_names = {""};
    std::vector<std::string> prefixes = {""};
    std::vector<std::string> suffixes = {""};
    std::vector<std::string> io_formats = {"posix"};
    pressio_options results;
    pressio_options defaults;
		int write_inputs = 1;
		int write_outputs = 1;
    double duration = 0.0;
    std::vector<pressio_io> io_modules = {std::shared_ptr<libpressio_io_plugin>(io_plugins().build("posix"))};

};


static pressio_register metrics_external_plugin(metrics_plugins(), "external", [](){ return compat::make_unique<external_metric_plugin>(); });
} }
