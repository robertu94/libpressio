#include <cmath>
#include <string>
#include <vector>
#include <utility>
#include <sstream>
#include <iterator>
#include <sys/wait.h>
#include <unistd.h>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/io/pressio_io.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/cpp/io.h"
#include "libpressio_ext/compat/std_compat.h"

using namespace std::literals;

namespace {
    enum extern_proc_error_codes {
      success=0,
      pipe_error=1,
      fork_error=2,
      exec_error=3,
      format_error=4
    };
    struct extern_proc_results {
      std::string proc_stdout; //stdout from the command
      std::string proc_stderr; //stdin from the command
      int return_code = 0; //the return code from the external process
      int error_code = success; //used to report errors with run_command
    };

    extern_proc_results run_command(std::string const& full_command) {
      extern_proc_results results;

      //create the pipe for stdout
      int stdout_pipe_fd[2];
      if(int ec = pipe(stdout_pipe_fd)) {
        results.return_code = ec;
        results.error_code = pipe_error;
        return results;
      }

      //create the pipe for stderr
      int stderr_pipe_fd[2];
      if(int ec = pipe(stderr_pipe_fd)) {
        results.return_code = ec;
        results.error_code = pipe_error;
        return results;
      }

      //run the program
      int child = fork();
      switch (child) {
        case -1:
          results.return_code = -1;
          results.error_code = fork_error;
          break;
        case 0:
          //in the child process
          {
          close(0);
          close(stdout_pipe_fd[0]);
          close(stderr_pipe_fd[0]);
          dup2(stdout_pipe_fd[1], 1);
          dup2(stderr_pipe_fd[1], 2);
          std::istringstream command_stream(full_command);
          std::vector<std::string> args_mem(
              std::istream_iterator<std::string>{command_stream},
              std::istream_iterator<std::string>());
          std::vector<char*> args;
          std::transform(std::begin(args_mem), std::end(args_mem),
              std::back_inserter(args), [](std::string const& s){return const_cast<char*>(s.c_str());});
          args.push_back(nullptr);
          execvp(args.front(), args.data());
          printf("%s\n", args.front());
          perror(" failed to exec process");
          //exit if there was an error
          exit(-1);
          break;
          }
        default:
          //in the parent process

          //close the unused parts of pipes
          close(stdout_pipe_fd[1]);
          close(stderr_pipe_fd[1]);

          int status = 0;
          char buffer[2048];
          std::ostringstream stdout_stream;
          std::ostringstream stderr_stream;
          do {
            //read the stdout[0]
            int nread;
            while((nread = read(stdout_pipe_fd[0], buffer, 2048)) > 0) {
              stdout_stream.write(buffer, nread);
            }
            
            //read the stderr[0]
            while((nread = read(stderr_pipe_fd[0], buffer, 2048)) > 0) {
              stderr_stream.write(buffer, nread);
            }

            //wait for the child to complete
            waitpid(child, &status, 0);
          } while (not WIFEXITED(status));

          results.proc_stdout = stdout_stream.str();
          results.proc_stderr = stderr_stream.str();
          results.return_code = WEXITSTATUS(status);
      }


      return results;
    }
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
      set(opt, "external:command", command);
      set(opt, "external:io_format", io_formats);
      return opt;
    }

    int set_options(pressio_options const& opt) override {
      get(opt, "external:command", &command);
      if(get(opt,"external:io_format", &io_formats) == pressio_options_key_set) {
        pressio library;
        io_modules.clear();
        std::transform(std::begin(io_formats),
                       std::end(io_formats),
                       std::back_inserter(io_modules),
                       [&library,&opt](std::string const& format) {
                       auto io_module =  library.get_io(format); 
                       io_module->set_options(opt);
                       return io_module;
                       }
            );
      }
      return 0;
    }

    struct pressio_options get_metrics_results() const override {
      if(results.size() == 0) {
        pressio_options ret;
        auto default_result = run_command(command);
        parse_result(default_result, ret);
        return ret;
      }
      return results;
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

      return cloned;
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
            parse_v1(stdout_stream, proc_results, results);
            return api_version;
          default:
            (void)0;
        }
      } catch(...) {} //swallow all errors and set error information

      results.clear();
      set(results, "external:error_code", (int)format_error);
      set(results, "external:return_code", 0);
      set(results, "external:stderr", "");
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
    }

    std::string build_command(std::vector<std::pair<std::string,std::string>> const& filenames, std::vector<std::reference_wrapper<const pressio_data>> const& input_datasets) const {
      std::ostringstream ss;
      ss << command;
      ss << " --api 3";
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
        auto const& decomp_path = filenames[i].first;
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
        std::string input_fd_name = get_or(prefixes, i).value_or("") + ".pressioinXXXXXX"s + get_or(suffixes, i).value_or(""s);
        int input_fd = mkstemps(&input_fd_name[0], get_or(suffixes, i).value_or("").size());
        io_modules[i]->set_options({{"io:path", std::string(input_fd_name)}});
        io_modules[i]->write(&input_data[i].get());

        //write decompressed data to a temporary file
        std::string output_fd_name = get_or(prefixes, i).value_or(""s) + ".pressiooutXXXXXX"s + get_or(suffixes, i).value_or("");
        int decompressed_fd = mkstemps(&output_fd_name[0], get_or(suffixes,i).value_or(""s).size());
        io_modules[i]->set_options({{"io:path", std::string(output_fd_name)}});
        io_modules[i]->write(&decompressed_data[i].get());
        
        filenames.emplace_back(input_fd_name, output_fd_name);
        fds.emplace_back(input_fd);
        fds.emplace_back(decompressed_fd);
      }

      //get the defaults
      auto default_result = run_command(command);
      parse_result(default_result, this->defaults);

      //build the command
      std::string full_command = build_command(filenames, input_data);

      //run the external program
      auto result = run_command(full_command);

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
    std::string workdir;
    std::vector<std::string> field_names;
    std::vector<std::string> prefixes;
    std::vector<std::string> suffixes;
    std::vector<std::string> io_formats = {"posix"};
    pressio_options results;
    pressio_options defaults;
    std::vector<pressio_io> io_modules = {std::shared_ptr<libpressio_io_plugin>(io_plugins().build("posix"))};

};


static pressio_register X(metrics_plugins(), "external", [](){ return compat::make_unique<external_metric_plugin>(); });
