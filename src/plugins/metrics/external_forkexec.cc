#include "external_launch.h"
#include <memory>
#include <sstream>
#include <unistd.h>
#include <iterator>
#include <sys/wait.h>
#include "std_compat/memory.h"

struct external_forkexec: public libpressio_launch_plugin {
extern_proc_results launch(std::string const& full_command, std::string const& workdir) const override {
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
          close(stdout_pipe_fd[0]);
          close(stderr_pipe_fd[0]);
          dup2(stdout_pipe_fd[1], 1);
          dup2(stderr_pipe_fd[1], 2);

          int chdir_status = chdir(workdir.c_str());
          if(chdir_status == -1) {
            perror(" failed to change to the specified directory");
            exit(-2);
          }

          std::istringstream command_stream(full_command);
          std::vector<std::string> args_mem(
              std::istream_iterator<std::string>{command_stream},
              std::istream_iterator<std::string>());
          std::vector<char*> args;
          std::transform(std::begin(args_mem), std::end(args_mem),
              std::back_inserter(args), [](std::string const& s){return const_cast<char*>(s.c_str());});
          args.push_back(nullptr);
          if(args.front() != nullptr) {
            execvp(args.front(), args.data());
            fprintf(stdout, "external:api=5");
            perror("failed to exec process");
            //exit if there was an error
            fprintf(stderr, " %s\n", args.front());
          } else {
            fprintf(stdout, "external:api=5");
            fprintf(stderr, "no process set");
          }
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
  const char* prefix() const override {
    return "forkexec";
  }

  
  std::unique_ptr<libpressio_launch_plugin> clone() const override {
    return compat::make_unique<external_forkexec>(*this);
  }
};

static pressio_register launch_forkexec_plugin(launch_plugins(), "forkexec", [](){ return compat::make_unique<external_forkexec>();});
