#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <unistd.h>
#include <chrono>
#include <ratio>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/io/pressio_io.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/compat/memory.h"
#include "libpressio_ext/compat/language.h"

#include "external_launch.h"



pressio_registry<std::unique_ptr<libpressio_launch_plugin>>& launch_plugins() {
  static pressio_registry<std::unique_ptr<libpressio_launch_plugin>> registry;
  return registry;
}

class external_metric_plugin : public libpressio_metrics_plugin {

  public:

    void begin_compress(const struct pressio_data * input, struct pressio_data const * ) override {
      input_data = pressio_data::clone(*input);
    }
    void end_decompress(struct pressio_data const*, struct pressio_data const* output, int ) override {
      std::vector<std::reference_wrapper<const pressio_data>> input_datasets{input_data};
      std::vector<std::reference_wrapper<const pressio_data>> output_datasets{*output};
      run_external(input_datasets, output_datasets);
    }

    struct pressio_options get_options() const override {
      auto opt = pressio_options();
      set_meta(opt, "external:launch_method", launch_method, launcher);
      set_meta_many(opt, "external:io_format", io_formats, io_modules);
      set(opt, "external:command", command);
      set(opt, "external:suffix", suffixes);
      set(opt, "external:prefix", prefixes);
      set(opt, "external:fieldnames", field_names);
      set(opt, "external:workdir", workdir);
      set(opt, "external:config_name", config_name);
      return opt;
    }

    int set_options(pressio_options const& opt) override {
      get_meta(opt, "external:launch_method", launch_plugins(), launch_method, launcher);
      get(opt, "external:command", &command);
      get(opt, "external:suffix", &suffixes);
      get(opt, "external:prefix", &prefixes);
      get(opt, "external:fieldnames", &field_names);
      get(opt, "external:workdir", &workdir);
      get(opt, "external:config_name", &config_name);
      get_meta_many(opt, "external:io_format", io_plugins(), io_formats, io_modules);
      return 0;
    }

    struct pressio_options get_metrics_results() const override {
      if(results.size() == 0) {
        pressio_options ret;
        auto default_result = launcher->launch(command, workdir);
        parse_result(default_result, ret);
        return ret;
      }
      return results;
    }

    void set_name_impl(std::string const& new_name) override {
      for (size_t i = 0; i < std::min(io_modules.size(), field_names.size()); ++i) {
        io_modules[i]->set_name(new_name + "/" + field_names[i]);
      }
      launcher->set_name(new_name + "/" + launcher->prefix());
    }

    std::unique_ptr<libpressio_metrics_plugin> clone() override {
      auto cloned = compat::make_unique<external_metric_plugin>();
      cloned->input_data = this->input_data;
      cloned->command = this->command;
      cloned->io_formats = this->io_formats;
      cloned->results = this->results;
      cloned->io_modules.clear();
      std::transform(std::begin(this->io_modules),
          std::end(this->io_modules),
          std::back_inserter(cloned->io_modules),
          std::mem_fn(&libpressio_io_plugin::clone)
          );

      return RVO_MOVE(cloned);
    }

    const char* prefix() const override {
      return "external";
    }


  private:


    //returns the version number parsed, starts at 1, zero means error
    size_t api_version_number(std::istringstream& stdout_stream) const {
      std::string version_line;
      std::getline(stdout_stream, version_line);
      auto eq_pos = version_line.find('=') + 1;
      if(version_line.substr(0, eq_pos) == "external:api") {
        //report error
        return 0;
      }
      return stoull(version_line.substr(eq_pos));
    }

    size_t parse_result(extern_proc_results& proc_results, pressio_options& results) const
    {
      try{
        std::istringstream stdout_stream(proc_results.proc_stdout);
        size_t api_version = api_version_number(stdout_stream);
        switch(api_version) {
          case 1:
          case 2:
          case 3:
          case 4:
          case 5:
            parse_v1(stdout_stream, proc_results, results);
            return api_version;
          default:
            (void)0;
        }
      } catch(...) {} //swallow all errors and set error information

      results.clear();
      set(results, "external:error_code", (int)format_error);
      set(results, "external:return_code", 0);
      set(results, "external:stderr", proc_results.proc_stderr);
      set(results, "external:runtime", duration);
      return 0;
    }

    void parse_v1(std::istringstream& stdout_stream, extern_proc_results& input, pressio_options& results) const {
      results.clear();

      for (std::string line; std::getline(stdout_stream, line); ) {
        auto equal_pos = line.find('=');
        std::string name = "external:results:" + line.substr(0, equal_pos);
        std::string value_s = line.substr(equal_pos + 1);
        double value = std::stod(value_s);
        set(results, name, value);
      }
      set(results, "external:stderr", input.proc_stderr);
      set(results, "external:return_code", input.return_code);
      set(results, "external:error_code", input.return_code);
      set(results, "external:runtime", duration);
    }

    std::string build_command(std::vector<std::pair<std::string,std::string>> const& filenames, std::vector<std::reference_wrapper<const pressio_data>> const& input_datasets) const {
      std::ostringstream ss;
      ss << command;
      ss << " --api 5";
      ss << " --config_name " << config_name;
      auto format_arg = [this](size_t i, std::string const& arg) {
        std::ostringstream ss;
        if(i >= field_names.size() || field_names[i].empty()) {
          ss << " --" << arg << ' ';
        } else {
          ss << " --" << field_names[i] << '_' << arg << ' ';
        }
        return ss.str();
      };
      for (size_t i = 0; i < filenames.size(); ++i) {
        auto const& input_path = filenames[i].first;
        auto const& decomp_path = filenames[i].second;
        auto const& input_data = input_datasets[i].get();
        ss << format_arg(i, "input") << input_path;
        ss << format_arg(i, "decompressed") << decomp_path;
        ss << format_arg(i, "type");
        switch(input_data.dtype()) {
          case pressio_float_dtype: ss << "float"; break;
          case pressio_double_dtype: ss << "double"; break;
          case pressio_int8_dtype: ss << "int8"; break;
          case pressio_int16_dtype: ss << "int16"; break;
          case pressio_int32_dtype: ss << "int32"; break;
          case pressio_int64_dtype: ss << "int64"; break;
          case pressio_uint8_dtype: ss << "uint8"; break;
          case pressio_uint16_dtype:ss << "uint16"; break;
          case pressio_uint32_dtype:ss << "uint32"; break;
          case pressio_uint64_dtype:ss << "uint64"; break;
          case pressio_byte_dtype:ss << "byte"; break;
        }
        for (auto i : input_data.dimensions()) {
          ss << format_arg(i, "dim") << i;
        }
        
      }
      return ss.str();
    }

    void run_external(std::vector<std::reference_wrapper<const pressio_data>> const& input_data, std::vector<std::reference_wrapper<const pressio_data>> const& decompressed_data) {
      std::vector<int> fds;
      std::vector<std::pair<std::string,std::string>> filenames;
      auto get_or = [](std::vector<std::string> const& array, size_t index) -> compat::optional<std::string> {
        if(index < array.size()){
          return array[index];
        } else {
          return {};
        }
      };

      for (size_t i = 0; i < io_modules.size(); ++i) {
        //write uncompressed data to a temporary file
        std::string input_fd_name = get_or(prefixes, i).value_or("") + std::string(".pressioinXXXXXX") + get_or(suffixes, i).value_or("");
        int input_fd = mkstemps(&input_fd_name[0], get_or(suffixes, i).value_or("").size());
        char* resolved_input = realpath(input_fd_name.c_str(), nullptr);
        input_fd_name = resolved_input;
        free(resolved_input);
        io_modules[i]->set_options({{"io:path", std::string(input_fd_name)}});
        io_modules[i]->write(&input_data[i].get());

        //write decompressed data to a temporary file
        std::string output_fd_name = get_or(prefixes, i).value_or("") + std::string(".pressiooutXXXXXX") + get_or(suffixes, i).value_or("");
        int decompressed_fd = mkstemps(&output_fd_name[0], get_or(suffixes,i).value_or("").size());
        char* resolved_output = realpath(output_fd_name.c_str(), nullptr);
        output_fd_name = resolved_output;
        free(resolved_output);
        io_modules[i]->set_options({{"io:path", std::string(output_fd_name)}});
        io_modules[i]->write(&decompressed_data[i].get());
        
        filenames.emplace_back(input_fd_name, output_fd_name);
        fds.emplace_back(input_fd);
        fds.emplace_back(decompressed_fd);
      }

      //get the defaults
      auto default_result = launcher->launch(command, workdir);
      parse_result(default_result, this->defaults);

      //build the command
      std::string full_command = build_command(filenames, input_data);

      //run the external program
      auto start_time = std::chrono::high_resolution_clock::now();
      auto result = launcher->launch(full_command, workdir);
      auto end_time = std::chrono::high_resolution_clock::now();
      duration = std::chrono::duration<double>(end_time - start_time).count();

      //parse the output
      size_t api_version =parse_result(result, this->results);

      //combine the results by setting defaults
      if(api_version >= 3) {
        for (auto const& default_v : defaults) {
          if (results.key_status(default_v.first) != pressio_options_key_set) {
            results.set(default_v.first, defaults.get(default_v.first));
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

    pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
    std::string command;
    std::string workdir = ".";
    std::string launch_method = "forkexec";
    std::string config_name = "external";
    std::unique_ptr<libpressio_launch_plugin> launcher = launch_plugins().build("forkexec");
    std::vector<std::string> field_names = {""};
    std::vector<std::string> prefixes = {""};
    std::vector<std::string> suffixes = {""};
    std::vector<std::string> io_formats = {"posix"};
    pressio_options results;
    pressio_options defaults;
    double duration = 0.0;
    std::vector<pressio_io> io_modules = {std::shared_ptr<libpressio_io_plugin>(io_plugins().build("posix"))};

};


static pressio_register metrics_external_plugin(metrics_plugins(), "external", [](){ return compat::make_unique<external_metric_plugin>(); });
