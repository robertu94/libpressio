#include "libpressio_ext/launch/external_launch.h"
#include <iterator>
#include <memory>
#include <sstream>
#include <mpi.h>
#include "pressio_compressor.h"
#include "std_compat/memory.h"

struct external_mpispawn: public libpressio_launch_plugin {
extern_proc_results launch(std::vector<std::string> const& full_command) const override {
      extern_proc_results results;

      std::vector<char*> args;
      for (auto const& command: commands) {
        args.push_back(const_cast<char*>(command.c_str()));
      }
      std::transform(std::begin(full_command), std::end(full_command),
          std::back_inserter(args), [](std::string const& s){return const_cast<char*>(s.c_str());});
      args.push_back(nullptr);

      MPI_Info info;
      MPI_Info_create(&info);
      if(not workdir.empty()) {
        MPI_Info_set(info, "wdir", workdir.c_str());
      }

      int error_code;
      MPI_Comm child;
      MPI_Comm_spawn(args.front(), args.data()+1, 1, info, 0,  MPI_COMM_SELF, &child, &error_code); 

      int status_code;
      MPI_Recv(&status_code, 1, MPI_INT, 0, 0, child, MPI_STATUS_IGNORE);

      int stdout_len;
      MPI_Recv(&stdout_len, 1, MPI_INT, 0, 0, child, MPI_STATUS_IGNORE);
      results.proc_stdout.resize(stdout_len);
      MPI_Recv(&results.proc_stdout[0], stdout_len, MPI_CHAR, 0, 0, child, MPI_STATUS_IGNORE);

      int stderr_len;
      MPI_Recv(&stderr_len, 1, MPI_INT, 0, 0, child, MPI_STATUS_IGNORE);
      results.proc_stderr.resize(stderr_len);
      MPI_Recv(&results.proc_stderr[0], stderr_len, MPI_CHAR, 0, 0, child, MPI_STATUS_IGNORE);

      MPI_Comm_free(&child);
      MPI_Info_free(&info);
      return results;
    }
  const char* prefix() const override {
    return "mpispawn";
  }

  int set_options(pressio_options const& options) override {
    get(options, "external:workdir", &workdir);
    get(options, "external:commands", &commands);
    return 0;
  }

  struct pressio_options get_configuration() const override {
    struct pressio_options options;
    set(options, "pressio:thread_safe", static_cast<int32_t>(pressio_thread_safety_multiple));
    set(options, "pressio:stability", "stable");
    return options;
  }


  pressio_options get_documentation_impl() const override {
    pressio_options options;
    set(options, "pressio:description", "spawn the child process using MPI_Comm_spawn on MPI_COMM_SELF");
    set(options, "external:workdir", "working directory for the child process");
    set(options, "external:commands", "list of strings passed to exec");
    return options;
  }


  pressio_options get_options() const override {
    pressio_options options;
    set(options, "external:workdir", workdir);
    set(options, "external:commands", commands);
    return options;
  }

  std::unique_ptr<libpressio_launch_plugin> clone() const override {
    return compat::make_unique<external_mpispawn>(*this);
  }

  std::string workdir=".";
  std::vector<std::string> commands;
};

static pressio_register launch_spawn_plugin(launch_plugins(), "mpispawn", [](){ return compat::make_unique<external_mpispawn>();});
