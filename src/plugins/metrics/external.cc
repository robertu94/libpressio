#include <cmath>
#include <string>
#include <sstream>
#include <iterator>
#include <sys/wait.h>
#include <unistd.h>
#include "pressio_data.h"
#include "pressio_options.h"
#include "libpressio_ext/io/posix.h"
#include "libpressio_ext/cpp/data.h"
#include "libpressio_ext/cpp/metrics.h"
#include "libpressio_ext/cpp/options.h"
#include "libpressio_ext/cpp/pressio.h"
#include "libpressio_ext/compat/std_compat.h"

namespace {
    enum extern_proc_error_codes {
      success=0,
      pipe_error=1,
      fork_error=2,
      exec_error=3,
      format_error=4
    };
    struct extern_proc_results {
      std::stringstream stdout_stream; //stdout from the command
      std::stringstream stderr_stream; //stdin from the command
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
          do {
            //read the stdout[0]
            int nread;
            while((nread = read(stdout_pipe_fd[0], buffer, 2048)) > 0) {
              results.stdout_stream.write(buffer, nread);
            }
            
            //read the stderr[0]
            while((nread = read(stderr_pipe_fd[0], buffer, 2048)) > 0) {
              results.stderr_stream.write(buffer, nread);
            }

            //wait for the child to complete
            waitpid(child, &status, 0);
          } while (not WIFEXITED(status));

          results.stdout_stream.seekg(0);
          results.stderr_stream.seekg(0);
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
      run_external(input_data, *output);
    }

    struct pressio_options get_metrics_options() const override {
      auto opt = pressio_options();
      opt.set("external:command", command);
      return opt;
    }

    int set_metrics_options(pressio_options const& opt) override {
      const char* tmp = nullptr;
      if(opt.get("external:command", &tmp) == pressio_options_key_set) {
        command = tmp;
      }
      free((void*)tmp);
      return 0;
    }

    struct pressio_options get_metrics_results() const override {
      return results;
    }


  private:


    //returns the version number parsed, starts at 1, zero means error
    size_t api_version_number(extern_proc_results& results) {
      std::string version_line;
      std::getline(results.stdout_stream, version_line);
      auto eq_pos = version_line.find('=') + 1;
      if(version_line.substr(0, eq_pos) == "external:api") {
        //report error
        return 0;
      }
      return stoull(version_line.substr(eq_pos));
    }

    void parse_result(extern_proc_results& results) {
      try{
        size_t api_version = api_version_number(results);
        switch(api_version) {
          case 1:
            parse_v1(results);
            return;
          default:
            (void)0;
        }
      } catch(...) {} //swallow all errors and set error information

      this->results.clear();
      this->results.set("external:error_code", (int)format_error);
      this->results.set("external:return_code", 0);
      this->results.set("external:stderr", "");
    }

    void parse_v1(extern_proc_results& input) {
      results.clear();

      for (std::string line; std::getline(input.stdout_stream, line); ) {
        auto equal_pos = line.find('=');
        std::string name = "external:results:" + line.substr(0, equal_pos);
        std::string value_s = line.substr(equal_pos + 1);
        double value = std::stod(value_s);
        results.set(name, value);
      }
      results.set("external:stderr", input.stderr_stream.str());
      results.set("external:return_code", input.return_code);
      results.set("external:error_code", input.return_code);
    }

    std::string build_command(std::string const& input_path, std::string const& decomp_path, pressio_data const& input_data) const {
      std::ostringstream ss;
      ss << command;
      ss << " --api 1";
      ss << " --input " << input_path;
      ss << " --decompressed " << decomp_path;
      ss << " --type ";
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
        ss << " --dim " << i;
      }
      return ss.str();
    }

    void run_external(pressio_data const& input_data, pressio_data const& decompressed_data) {

      //write uncompressed data to a temporary file
      char input_fd_name[] = {'.', 'p', 'r', 'e', 's', 's','i', 'o','i', 'n', 'X', 'X', 'X', 'X', 'X', 'X', 0};
      int input_fd = mkstemp(input_fd_name);
      pressio_io_data_write(&input_data, input_fd);

      //write decompressed data to a temporary file
      char output_fd_name[] = {'.', 'p', 'r', 'e', 's', 's','i', 'o','o', 'u','t', 'X', 'X', 'X', 'X', 'X', 'X', 0};
      int decompressed_fd = mkstemp(output_fd_name);
      pressio_io_data_write(&decompressed_data, decompressed_fd);

      //build the command
      std::string full_command = build_command(input_fd_name, output_fd_name, input_data);

      //run the external program
      auto result = run_command(full_command);

      //parse the output
      parse_result(result);

      //delete the temporary files
      close(input_fd);
      close(decompressed_fd);
      unlink(input_fd_name);
      unlink(output_fd_name);
    }

    

    pressio_data input_data = pressio_data::empty(pressio_byte_dtype, {});
    std::string command;
    pressio_options results;

};


static pressio_register X(metrics_plugins(), "external", [](){ return compat::make_unique<external_metric_plugin>(); });

