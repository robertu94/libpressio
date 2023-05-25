#include "libpressio_ext/launch/external_launch.h"
#include <memory>
#include <sstream>
#include <unistd.h>
#include <iterator>
#include <sys/wait.h>
#include <errno.h>
#include "pressio_compressor.h"
#include "std_compat/memory.h"

struct external_forkexec: public libpressio_launch_plugin {
extern_proc_results launch_impl(std::vector<std::string> const& full_command) const override {
      extern_proc_results results;

      if(commands.size() == 0) {
          results.return_code = 0;
          results.error_code = fork_error;
          results.proc_stderr = "command not set";
          return results;
      }

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
          close(STDIN_FILENO);
          close(stdout_pipe_fd[0]);
          close(stderr_pipe_fd[0]);
          dup2(stdout_pipe_fd[1], 1);
          dup2(stderr_pipe_fd[1], 2);

          int chdir_status = chdir(workdir.c_str());
          if(chdir_status == -1) {
            perror(" failed to change to the specified directory");
            exit(-2);
          }

          std::vector<char*> args;
          args.reserve(commands.size());
          for(auto const& command: commands) {
            args.push_back(const_cast<char*>(command.c_str()));
          }
          std::transform(std::begin(full_command), std::end(full_command),
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
          bool stdout_closed = false, stderr_closed = false;

          while(true) {
            int ready, nfds = 0;
            ssize_t nread;
            fd_set readfds, writefds, exceptfds;

            FD_ZERO(&readfds);
            FD_ZERO(&writefds);
            FD_ZERO(&exceptfds);
            if(not stdout_closed) {
              FD_SET(stdout_pipe_fd[0], &readfds);
              nfds = std::max(stdout_pipe_fd[0], nfds);
            }
            if(not stderr_closed) {
              FD_SET(stderr_pipe_fd[0], &readfds);
              nfds = std::max(stderr_pipe_fd[0], nfds);
            }

            if(stdout_closed && stderr_closed) {
              break;
            }

            ready = select(nfds+1, &readfds, &writefds, &exceptfds, nullptr);

            if(ready == -1 && errno == EINTR) {
              continue;
            }

            if(!stdout_closed && FD_ISSET(stdout_pipe_fd[0], &readfds)) {
              nread = read(stdout_pipe_fd[0], buffer, sizeof buffer);
              if(nread > 0) {
                stdout_stream.write(buffer, nread);
              } else if (nread == 0) {
                stdout_closed = true;
              }
            }

            if(!stderr_closed && FD_ISSET(stderr_pipe_fd[0], &readfds)) {
              nread = read(stderr_pipe_fd[0], buffer, sizeof buffer);
              if(nread > 0) {
                stderr_stream.write(buffer, nread);
              } else if(nread == 0)  {
                stderr_closed = true;
              }
            }
          }

          waitpid(child, &status, 0);
          close(stdout_pipe_fd[0]);
          close(stderr_pipe_fd[0]);
//          do {
//            //read the stdout[0]
//            int nread;
//            while((nread = read(stdout_pipe_fd[0], buffer, 2048)) > 0) {
//              stdout_stream.write(buffer, nread);
//            }
//            
//            //read the stderr[0]
//            while((nread = read(stderr_pipe_fd[0], buffer, 2048)) > 0) {
//              stderr_stream.write(buffer, nread);
//            }
//
//            //wait for the child to complete
//            waitpid(child, &status, 0);
//          } while (not WIFEXITED(status));

          results.proc_stdout = stdout_stream.str();
          results.proc_stderr = stderr_stream.str();
          results.return_code = WEXITSTATUS(status);
      }


      return results;
    }
  const char* prefix() const override {
    return "forkexec";
  }

  int set_options_impl(pressio_options const& options) override {
    get(options, "external:workdir", &workdir);
    std::string command;
    if(get(options, "external:commands", &command) == pressio_options_key_set) {
        commands = {command};
    } else {
        get(options, "external:commands", &commands);
    }
    return 0;
  }

  struct pressio_options get_configuration_impl() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", pressio_thread_safety_multiple);
    set(options, "pressio:stability", "stable");
    return options;
  }

  pressio_options get_documentation_impl() const override {
    pressio_options options;
    set(options, "pressio:description", "spawn the child process using fork+exec");
    set(options, "external:workdir", "working directory for the child process");
    set(options, "external:commands", "list of strings passed to exec");
    return options;
  }

  pressio_options get_options_impl() const override {
    pressio_options options;
    set(options, "external:workdir", workdir);
    set(options, "external:commands", commands);
    return options;
  }

  
  std::unique_ptr<libpressio_launch_plugin> clone() const override {
    return compat::make_unique<external_forkexec>(*this);
  }

  std::string workdir=".";
  std::vector<std::string> commands;
};

static pressio_register launch_forkexec_plugin(launch_plugins(), "forkexec", [](){ return compat::make_unique<external_forkexec>();});
